# Polymer Processing - Optical Profilometry

Tool for analyzing polymer surfaces using optical profilometry data.

## Quick Start

### 1. Build Docker Image
```bash
cd /path/to/dockerfile/directory
docker build -t polymer-processing .
```

### 2. Run Container
```bash
./run_docker.sh
```
*When prompted, provide the path to your raw data directory (will be mounted to `/data`)*

### 3. Create Dataset
```bash
cd /code/polymer
./create_dataset.sh
```
*This generates a tfrecords dataset path*

### 4. Start Jupyter
```bash
cd /code
./start_jupyter.sh
```
*Note the port number (e.g., 8889)*

**In a new terminal on your local machine:**
```bash
ssh -i ~/.ssh/your_key -p YOUR_PORT -L 8889:localhost:8889 user@your.server.ip
```
Then go to `localhost:8889` and copy the token to access Jupyter.

### 5. Train Model
- Open `TrainPolymer` notebook
- Set dataset path in top cell:
  ```python
  dataset_path = '/data/tfrecords_pe'
  ```
- Run notebook to get checkpoint path like:
  ```
  /data/experiments/vae_baseline_20250905_123021
  ```

### 6. Explore Results
- Open `LatentsPolymer` notebook
- Update paths:
  ```python
  viz = VAEExplorer(
      dataset_path='/data/tfrecords_pe',
      checkpoint_path='/data/experiments/vae_baseline_20250905_123021/model_checkpoint'
  )
  viz.load_data(50)
  viz.compute_and_plot_tsne(perplexity=30)
  browser = viz.create_image_browser()
  display(browser)
  ```

## Workflow Summary
1. Build → 2. Run → 3. Create Dataset → 4. Start Jupyter → 5. Train → 6. Explore
