import onnx
import numpy as np
from onnx import helper, TensorProto, ValueInfoProto
from collections import defaultdict, deque

def get_tensor_shape_from_graph(graph, tensor_name):
    """
    Get the shape of a tensor from graph initializers or value_info
    """
    # Check initializers first
    for init in graph.initializer:
        if init.name == tensor_name:
            return list(init.dims)
    
    # Check value_info
    for value_info in graph.value_info:
        if value_info.name == tensor_name:
            if value_info.type.tensor_type.shape.dim:
                return [d.dim_value for d in value_info.type.tensor_type.shape.dim]
    
    # Check inputs
    for inp in graph.input:
        if inp.name == tensor_name:
            if inp.type.tensor_type.shape.dim:
                return [d.dim_value for d in inp.type.tensor_type.shape.dim]
                
    return None

def infer_num_channels_from_conv(graph):
    """
    Infer number of channels from the Conv layer before InstanceNorm
    """
    # skip first conv
    idx = 0
    for node in graph.node:
        idx += 1
        if node.op_type == "Conv" and idx > 1:
            # Get weight tensor shape
            if len(node.input) > 1:
                weight_name = node.input[1]
                weight_shape = get_tensor_shape_from_graph(graph, weight_name)
                if weight_shape:
                    return weight_shape[0]  # Output channels
    return 32  # Default fallback

def reshape_parameter_tensor(graph, tensor_name, new_shape):
    """
    Reshape a parameter tensor (initializer) to new dimensions
    """
    for i, initializer in enumerate(graph.initializer):
        if initializer.name == tensor_name:
            # Get the current tensor data
            if initializer.data_type == TensorProto.FLOAT:
                # Convert to numpy array
                if initializer.raw_data:
                    tensor_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
                else:
                    tensor_data = np.array(initializer.float_data, dtype=np.float32)
                
                # Reshape to new dimensions
                reshaped_data = tensor_data.reshape(new_shape)
                
                # Create new initializer
                new_initializer = helper.make_tensor(
                    name=tensor_name,
                    data_type=TensorProto.FLOAT,
                    dims=new_shape,
                    vals=reshaped_data.flatten().tolist()
                )
                
                # Replace the old initializer
                graph.initializer[i].CopyFrom(new_initializer)
                print(f"  Reshaped {tensor_name}: {initializer.dims} -> {new_shape}")
                return True
    
    return False

def find_instancenorm_block(graph, instancenorm_node):
    """
    Find the complete InstanceNorm block and all nodes to replace
    """
    nodes_to_remove = []
    
    # Start from InstanceNorm
    current_node = instancenorm_node
    nodes_to_remove.append(current_node)
    
    # Find input Reshape
    input_tensor = current_node.input[0]
    input_reshape = None
    for node in graph.node:
        if node.output and node.output[0] == input_tensor and node.op_type == "Reshape":
            input_reshape = node
            break
    
    if input_reshape:
        nodes_to_remove.append(input_reshape)
        original_input = input_reshape.input[0]
    else:
        original_input = input_tensor
    
    # Follow the chain forward: InstanceNorm -> Reshape -> Mul -> Add
    current_output = current_node.output[0]
    scale_param = None
    bias_param = None
    
    # Find output Reshape
    output_reshape = None
    for node in graph.node:
        if node.input and node.input[0] == current_output and node.op_type == "Reshape":
            output_reshape = node
            break
    
    if output_reshape:
        nodes_to_remove.append(output_reshape) 
        current_output = output_reshape.output[0]
    
    # Find Mul
    mul_node = None
    for node in graph.node:
        if node.input and node.input[0] == current_output and node.op_type == "Mul":
            mul_node = node
            break
    
    if mul_node:
        nodes_to_remove.append(mul_node)
        scale_param = mul_node.input[1] if len(mul_node.input) > 1 else None
        current_output = mul_node.output[0]
    
    # Find Add
    add_node = None  
    for node in graph.node:
        if node.input and node.input[0] == current_output and node.op_type == "Add":
            add_node = node
            break
    
    if add_node:
        nodes_to_remove.append(add_node)
        bias_param = add_node.input[1] if len(add_node.input) > 1 else None
        final_output = add_node.output[0]
    else:
        final_output = current_output
    
    return nodes_to_remove, original_input, final_output, scale_param, bias_param

def topological_sort_graph(graph):
    """
    Topologically sort the nodes in the ONNX graph
    """
    # Build dependency graph
    node_inputs = {}  # node_name -> list of input tensor names
    tensor_producers = {}  # tensor_name -> node that produces it
    
    # Map nodes by name and build dependency info
    nodes_by_name = {}
    for i, node in enumerate(graph.node):
        node_name = node.name if node.name else f"node_{i}"
        nodes_by_name[node_name] = node
        node_inputs[node_name] = list(node.input)
        
        # Map output tensors to their producing nodes
        for output in node.output:
            tensor_producers[output] = node_name
    
    # Get graph inputs (from graph.input)
    graph_input_names = {inp.name for inp in graph.input}
    
    # Build adjacency list for dependencies
    dependencies = defaultdict(set)  # node -> set of nodes it depends on
    
    for node_name, inputs in node_inputs.items():
        for input_tensor in inputs:
            if input_tensor in tensor_producers:
                dependencies[node_name].add(tensor_producers[input_tensor])
    
    # Topological sort using Kahn's algorithm
    in_degree = defaultdict(int)
    for node_name in nodes_by_name:
        in_degree[node_name] = len(dependencies[node_name])
    
    queue = deque([node for node, degree in in_degree.items() if degree == 0])
    sorted_nodes = []
    
    while queue:
        current = queue.popleft()
        sorted_nodes.append(nodes_by_name[current])
        
        # Update in-degrees of dependent nodes
        for node_name, deps in dependencies.items():
            if current in deps:
                in_degree[node_name] -= 1
                if in_degree[node_name] == 0:
                    queue.append(node_name)
    
    if len(sorted_nodes) != len(graph.node):
        print(f"Warning: Topological sort may have failed. Expected {len(graph.node)} nodes, got {len(sorted_nodes)}")
    
    return sorted_nodes

def replace_instancenorm_blocks_with_groupnorm(model_path, output_path, num_groups=8):
    """
    Replace InstanceNorm blocks with GroupNorm and ensure proper topological sorting
    """
    model = onnx.load(model_path)
    graph = model.graph
    
    # Find all InstanceNormalization nodes
    instancenorm_nodes = [node for node in graph.node if node.op_type == "InstanceNormalization"]
    
    if not instancenorm_nodes:
        print("No InstanceNormalization nodes found!")
        return
    
    print(f"Found {len(instancenorm_nodes)} InstanceNormalization blocks to replace")
    
    # Infer number of channels from model structure
    num_channels = infer_num_channels_from_conv(graph)
    print(f"Inferred number of channels: {num_channels}")
    
    all_nodes_to_remove = []
    nodes_to_add = []
    
    for i, node in enumerate(instancenorm_nodes):
        print(f"\n--- Processing block {i+1}: {node.name} ---")
        
        # Get epsilon
        epsilon = 1e-5
        for attr in node.attribute:
            if attr.name == "epsilon":
                epsilon = attr.f
        
        # Find complete block
        block_nodes, original_input, final_output, scale_param, bias_param = find_instancenorm_block(graph, node)
        
        print(f"Removing {len(block_nodes)} nodes:")
        for block_node in block_nodes:
            print(f"  - {block_node.op_type}")
        
        # Use InstanceNorm's scale/bias if Mul/Add don't provide them
        if not scale_param and len(node.input) > 1:
            scale_param = node.input[1]
        if not bias_param and len(node.input) > 2:
            bias_param = node.input[2]
        
        # Reshape scale and bias parameters to 1D for GroupNorm compatibility
        if scale_param:
            print(f"  Reshaping scale parameter: {scale_param}")
            reshape_parameter_tensor(graph, scale_param, [num_channels])
            
        if bias_param:
            print(f"  Reshaping bias parameter: {bias_param}")
            reshape_parameter_tensor(graph, bias_param, [num_channels])
        
        # Create GroupNorm inputs
        groupnorm_inputs = [original_input]
        if scale_param:
            groupnorm_inputs.append(scale_param)
        if bias_param:
            groupnorm_inputs.append(bias_param)
        
        # Create GroupNorm node
        groupnorm_node = helper.make_node(
            "GroupNormalization",
            inputs=groupnorm_inputs,
            outputs=[final_output], 
            name=f"GroupNorm_{i}",
            num_groups=num_groups,
            epsilon=epsilon
        )
        
        all_nodes_to_remove.extend(block_nodes)
        nodes_to_add.append(groupnorm_node)
        
        print(f"-> Replacing with GroupNorm (groups={num_groups}, epsilon={epsilon})")
    
    # Remove old nodes
    for node in all_nodes_to_remove:
        if node in graph.node:
            graph.node.remove(node)
    
    # Add new nodes  
    graph.node.extend(nodes_to_add)
    
    print(f"\n=== Topologically sorting graph ===")
    # Sort the graph topologically
    try:
        sorted_nodes = topological_sort_graph(graph)
        
        # Replace the graph nodes with sorted ones
        graph.ClearField('node')
        graph.node.extend(sorted_nodes)
        
        print(f"✅ Graph sorted: {len(sorted_nodes)} nodes in proper order")
        
    except Exception as e:
        print(f"⚠️  Warning during topological sort: {e}")
    
    # Validate and save
    try:
        onnx.checker.check_model(model)
        onnx.save(model, output_path)
        print(f"\n🎉 Successfully saved model to: {output_path}")
        print(f"   - Replaced {len(instancenorm_nodes)} InstanceNorm blocks")
        print(f"   - Reshaped parameters for Burn compatibility")
        print(f"   - Graph is topologically sorted")
        print(f"   - Ready for Burn import!")
        
    except Exception as e:
        print(f"\n⚠️  Model validation warning: {e}")
        onnx.save(model, output_path)
        print(f"Model saved to: {output_path} (check manually)")

def analyze_model_structure(model_path):
    """
    Analyze the model structure to understand normalization blocks
    """
    model = onnx.load(model_path)
    graph = model.graph
    
    print("=== Model Structure Analysis ===")
    
    # Group nodes by type
    node_types = {}
    for node in graph.node:
        if node.op_type not in node_types:
            node_types[node.op_type] = []
        node_types[node.op_type].append(node.name)
    
    print("Node types in model:")
    for op_type, names in sorted(node_types.items()):
        print(f"  {op_type}: {len(names)} nodes")
    
    print("\n=== Parameter Shapes ===")
    for init in graph.initializer:
        if any(keyword in init.name.lower() for keyword in ['scale', 'bias', 'weight', 'gamma', 'beta']):
            print(f"  {init.name}: {list(init.dims)}")
    
    print("\n=== Normalization Blocks ===")
    # Find InstanceNorm and surrounding operations
    for node in graph.node:
        if node.op_type == "InstanceNormalization":
            print(f"\nInstanceNorm: {node.name}")
            print(f"  Input: {node.input}")
            print(f"  Output: {node.output}")
            
            # Check what feeds into it
            input_name = node.input[0]
            for prev_node in graph.node:
                if len(prev_node.output) > 0 and prev_node.output[0] == input_name:
                    print(f"  <- Fed by: {prev_node.op_type} ({prev_node.name})")
            
            # Check what it feeds into
            output_name = node.output[0]
            for next_node in graph.node:
                if output_name in next_node.input:
                    print(f"  -> Feeds into: {next_node.op_type} ({next_node.name})")

def fix_topological_order(model_path, output_path):
    """
    Just fix topological ordering of an existing model
    """
    model = onnx.load(model_path)
    graph = model.graph
    
    print("Fixing topological order...")
    sorted_nodes = topological_sort_graph(graph)
    
    graph.ClearField('node')
    graph.node.extend(sorted_nodes)
    
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"✅ Fixed and saved to: {output_path}")

# Main execution
if __name__ == "__main__":
    # Update these paths for your model
    import sys
    input_model = sys.argv[1] 
    output_model = 'model_gn.onnx' 
    
    print("=== ONNX InstanceNorm to GroupNorm Converter ===\n")
    
    # Analyze original model
    print("1. Analyzing original model structure...")
    analyze_model_structure(input_model)
    
    print("\n" + "="*50)
    
    # Convert InstanceNorm blocks to GroupNorm
    print("2. Converting InstanceNorm blocks to GroupNorm...")
    replace_instancenorm_blocks_with_groupnorm(
        input_model, 
        output_model, 
        num_groups=8  # Change this value as needed
    )
    
    print("\n" + "="*50)
    
    # Analyze converted model
    print("3. Analyzing converted model...")
    analyze_model_structure(output_model)
    
    print(f"\n✨ Conversion complete!")
    print(f"   Original: {input_model}")
    print(f"   Converted: {output_model}")
    print(f"\nTry importing the converted model into Burn now!")
