// Model Version Management for Food Classification Dashboard

class ModelVersionManager {
  constructor(apiBase) {
    this.API_BASE = apiBase;
    this.availableModels = [];
    this.currentModelVersion = "unknown";
  }

  // Initialize model version manager
  init() {
    console.log("🔧 Initializing Model Version Manager...");
    this.loadModelVersions();
    this.loadCurrentModelInfo();

    // Refresh model info every 30 seconds
    setInterval(() => {
      this.loadCurrentModelInfo();
    }, 30000);

    console.log("✅ Model Version Manager initialized");
  }

  // Load available model versions
  async loadModelVersions() {
    try {
      const response = await fetch(`${this.API_BASE}/models/versions`);
      const data = await response.json();

      this.availableModels = data.available_models;
      this.currentModelVersion = data.current_active_version;

      this.updateModelVersionUI();
    } catch (error) {
      console.error("Error loading model versions:", error);
      this.showAlert("Failed to load model versions", "error");
    }
  }

  // Update the UI with model version information
  updateModelVersionUI() {
    // Update current model display
    const currentModelDisplay = document.getElementById(
      "current-model-version"
    );
    if (currentModelDisplay) {
      currentModelDisplay.textContent = `${
        this.currentModelVersion === "original"
          ? "Original"
          : "v" + this.currentModelVersion
      }`;
    }

    // Update model selector dropdown (for switching active model)
    const modelSelector = document.getElementById("model-version-selector");
    if (modelSelector) {
      modelSelector.innerHTML = "";

      // Add latest option
      const latestOption = document.createElement("option");
      latestOption.value = "latest";
      latestOption.textContent = "Latest";
      if (this.currentModelVersion === "latest") latestOption.selected = true;
      modelSelector.appendChild(latestOption);

      // Add versioned models
      this.availableModels.forEach((model) => {
        const option = document.createElement("option");
        option.value = model.version;

        if (model.version === "original") {
          option.textContent = `Original Model (${model.file_size_mb}MB)`;
        } else {
          option.textContent = `v${model.version} (${model.file_size_mb}MB)`;
        }

        if (this.currentModelVersion == model.version) option.selected = true;
        modelSelector.appendChild(option);
      });
    }

    // Update prediction model selector dropdown
    this.updatePredictionModelSelector();
  }

  // Update prediction model selector
  updatePredictionModelSelector() {
    const predictionSelector = document.getElementById(
      "prediction-model-selector"
    );
    if (predictionSelector) {
      predictionSelector.innerHTML = "";

      // Add current active model option
      const currentOption = document.createElement("option");
      currentOption.value = "current";
      currentOption.textContent = "Current Active Model";
      currentOption.selected = true;
      predictionSelector.appendChild(currentOption);

      // Add latest option
      const latestOption = document.createElement("option");
      latestOption.value = "latest";
      latestOption.textContent = "Latest Model";
      predictionSelector.appendChild(latestOption);

      // Add all available models
      this.availableModels.forEach((model) => {
        const option = document.createElement("option");
        option.value = model.version;

        if (model.version === "original") {
          option.textContent = `Original Model`;
        } else {
          option.textContent = `v${model.version}`;
        }

        predictionSelector.appendChild(option);
      });
    }

    // Update model versions table
    this.updateModelVersionsTable();
  }

  // Update the model versions table in the UI
  updateModelVersionsTable() {
    const tableBody = document.getElementById("model-versions-table-body");
    if (!tableBody) return;

    tableBody.innerHTML = "";

    this.availableModels.forEach((model) => {
      const row = document.createElement("tr");
      row.className = model.is_latest
        ? "latest-model"
        : model.is_original
        ? "original-model"
        : "";

      const versionDisplay =
        model.version === "original" ? "Original" : model.version;
      const createdDate =
        model.created_date === "Original Model"
          ? "Original Model"
          : new Date(model.created_date).toLocaleDateString();

      row.innerHTML = `
                <td>
                    <span class="version-badge ${
                      model.is_latest
                        ? "latest"
                        : model.is_original
                        ? "original"
                        : ""
                    }">${versionDisplay}</span>
                    ${
                      model.is_latest
                        ? '<span class="latest-indicator">LATEST</span>'
                        : ""
                    }
                    ${
                      model.is_original
                        ? '<span class="original-indicator">ORIGINAL</span>'
                        : ""
                    }
                </td>
                <td>${model.file_size_mb} MB</td>
                <td>${createdDate}</td>
                <td>${model.num_classes}</td>
                <td>
                    <button 
                        class="btn btn-sm ${
                          this.currentModelVersion == model.version
                            ? "btn-success"
                            : "btn-primary"
                        }"
                        onclick="window.modelVersionManager.switchModelVersion('${
                          model.version
                        }')"
                        ${
                          this.currentModelVersion == model.version
                            ? "disabled"
                            : ""
                        }
                    >
                        ${
                          this.currentModelVersion == model.version
                            ? "Active"
                            : "Switch"
                        }
                    </button>
                </td>
            `;

      tableBody.appendChild(row);
    });
  }

  // Switch to a different model version
  async switchModelVersion(version) {
    try {
      this.showLoading("model-switch-loading");

      const response = await fetch(
        `${this.API_BASE}/models/switch/${version}`,
        {
          method: "POST",
        }
      );

      if (response.ok) {
        const data = await response.json();
        this.showAlert(`Successfully switched to model ${version}`, "success");

        // Reload model versions and current model info
        await this.loadModelVersions();
        await this.loadCurrentModelInfo();
      } else {
        const error = await response.json();
        this.showAlert(`Failed to switch model: ${error.detail}`, "error");
      }
    } catch (error) {
      console.error("Error switching model version:", error);
      this.showAlert("Failed to switch model version", "error");
    } finally {
      this.hideLoading("model-switch-loading");
    }
  }

  // Load current model information
  async loadCurrentModelInfo() {
    try {
      const response = await fetch(`${this.API_BASE}/models/current`);
      const data = await response.json();

      // Update current model info display
      this.updateCurrentModelDisplay(data);
    } catch (error) {
      console.error("Error loading current model info:", error);
    }
  }

  // Update current model display with detailed info
  updateCurrentModelDisplay(modelInfo) {
    const container = document.getElementById("current-model-info");
    if (!container) return;

    container.innerHTML = `
            <div class="current-model-card">
                <div class="model-version">
                    <span class="version-label">Active Model:</span>
                    <span class="version-value">v${modelInfo.version}</span>
                    ${
                      modelInfo.version === "latest"
                        ? '<span class="latest-badge">LATEST</span>'
                        : ""
                    }
                </div>
                <div class="model-details">
                    <div class="detail-item">
                        <span class="detail-label">Classes:</span>
                        <span class="detail-value">${
                          modelInfo.num_classes
                        }</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Size:</span>
                        <span class="detail-value">${
                          modelInfo.model_size_mb
                        } MB</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Predictions:</span>
                        <span class="detail-value">${
                          modelInfo.predictions_made
                        }</span>
                    </div>
                </div>
            </div>
        `;
  }

  // Add model version selector to prediction interface
  addModelSelectorToPrediction() {
    const predictionContainer = document.getElementById("prediction-interface");
    if (!predictionContainer) return;

    const selectorHTML = `
            <div class="model-selector-container">
                <label for="prediction-model-selector">Use Model Version:</label>
                <select id="prediction-model-selector" onchange="window.modelVersionManager.handlePredictionModelChange()">
                    <option value="current">Current Active Model</option>
                    <option value="latest">Latest Model</option>
                </select>
                <small class="model-selector-help">Select which model version to use for predictions</small>
            </div>
        `;

    predictionContainer.insertAdjacentHTML("beforeend", selectorHTML);
  }

  // Handle model change for predictions
  handlePredictionModelChange() {
    const selector = document.getElementById("prediction-model-selector");
    const selectedVersion = selector.value;

    if (selectedVersion !== "current") {
      // Switch model for predictions
      this.switchModelVersion(selectedVersion);
    }
  }

  // Utility methods
  showAlert(message, type) {
    // Use dashboard's showAlert if available, otherwise create own
    if (window.dashboard && window.dashboard.showAlert) {
      window.dashboard.showAlert(message, type);
    } else {
      console.log(`${type.toUpperCase()}: ${message}`);
    }
  }

  showLoading(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
      container.innerHTML =
        '<div class="loading"></div> <span style="color: rgba(255,255,255,0.8);">Processing...</span>';
      container.classList.remove("hidden");
    }
  }

  hideLoading(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
      container.innerHTML = "";
      container.classList.add("hidden");
    }
  }
}

// Global instance - will be initialized by dashboard
window.modelVersionManager = null;
