.PHONY: help setup download-model clean

# Directory where the model will be stored
MODEL_DIR := joint-disfluency-detector-and-parser/best_models
MODEL_FILE := swbd_fisher_bert_Edev.0.9078.pt
MODEL_URL := https://github.com/pariajm/joint-disfluency-detector-and-parser/releases/download/v1.0.0/$(MODEL_FILE)

help:
	@echo "Available targets:"
	@echo "  setup           - Create required directories"
	@echo "  download-model  - Download the pre-trained model"
	@echo "  clean           - Remove downloaded model files"

setup:
	@echo "Creating model directory..."
	@mkdir -p $(MODEL_DIR)

# Download the pre-trained model
download-model: setup
	@if [ ! -f "$(MODEL_DIR)/$(MODEL_FILE)" ]; then \
		echo "Downloading $(MODEL_FILE)..."; \
		curl -L $(MODEL_URL) -o "$(MODEL_DIR)/$(MODEL_FILE)"; \
		echo "Model downloaded to $(MODEL_DIR)/$(MODEL_FILE)"; \
	else \
		echo "Model already exists at $(MODEL_DIR)/$(MODEL_FILE)"; \
	fi

clean:
	@echo "Removing downloaded model..."
	@rm -f "$(MODEL_DIR)/$(MODEL_FILE)"
