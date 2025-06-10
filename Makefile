sync:
	@uv sync
	@uv pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
	@uv pip install mmengine