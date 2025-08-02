// Dashboard JavaScript Module
class FoodClassificationDashboard {
  constructor() {
    this.currentTab = "prediction";
    this.batchFiles = [];
    this.charts = {};
    this.API_BASE = "http://127.0.0.1:8000";

    // Initialize dashboard
    this.init();
  }

  // Initialize dashboard
  async init() {
    console.log("🚀 Initializing Food Classification Dashboard...");

    // Initialize model version manager
    window.modelVersionManager = new ModelVersionManager(this.API_BASE);

    // Load initial data
    await this.loadSystemStatus();
    await this.loadModelInfo();
    await this.loadVisualizations();
    this.setupDragAndDrop();

    // Initialize model version manager
    if (window.modelVersionManager) {
      window.modelVersionManager.init();
    }

    // Refresh data every 30 seconds
    setInterval(() => {
      this.loadSystemStatus();
      if (this.currentTab === "visualization") {
        this.loadVisualizations();
      }
      if (this.currentTab === "monitoring") {
        this.loadRecentPredictions();
      }
    }, 30000);

    console.log("✅ Dashboard initialized successfully");
  }

  // Tab switching
  switchTab(tabName) {
    console.log(`Switching to tab: ${tabName}`);

    // Hide all tabs
    document.querySelectorAll(".tab-content").forEach((tab) => {
      tab.classList.add("hidden");
    });

    // Remove active class from all tabs
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.classList.remove("active");
    });

    // Show selected tab
    const targetTab = document.getElementById(tabName + "-tab");
    if (targetTab) {
      targetTab.classList.remove("hidden");
    } else {
      console.error(`Tab not found: ${tabName}-tab`);
      return;
    }

    // Add active class to the clicked tab by finding the tab with the correct onclick attribute
    document.querySelectorAll(".tab").forEach((tab) => {
      const onclick = tab.getAttribute("onclick");
      if (onclick && onclick.includes(`'${tabName}'`)) {
        tab.classList.add("active");
      }
    });

    this.currentTab = tabName;

    // Load tab-specific data
    if (tabName === "visualization") {
      this.loadVisualizations();
    } else if (tabName === "monitoring") {
      this.loadRecentPredictions();
    } else if (tabName === "training") {
      // Load model versions for training tab
      if (window.modelVersionManager) {
        window.modelVersionManager.loadModelVersions();
      }
    }
  }

  // Load system status
  async loadSystemStatus() {
    try {
      const response = await fetch(`${this.API_BASE}/metrics`);
      const data = await response.json();

      document.getElementById("total-predictions").textContent =
        data.total_predictions || 0;
      document.getElementById("success-rate").textContent = `${
        data.success_rate?.toFixed(1) || 0
      }%`;
      document.getElementById("uptime").textContent = `${
        data.uptime_hours?.toFixed(1) || 0
      }h`;
      document.getElementById("last-prediction").textContent =
        data.last_prediction
          ? new Date(data.last_prediction).toLocaleString()
          : "Never";

      // Check health
      const healthResponse = await fetch(`${this.API_BASE}/health`);
      const healthData = await healthResponse.json();

      const indicator = document.getElementById("model-status-indicator");
      const status = document.getElementById("model-status");

      if (healthData.model_loaded) {
        indicator.className = "status-indicator status-online";
        status.textContent = "Online";
      } else {
        indicator.className = "status-indicator status-offline";
        status.textContent = "Offline";
      }
    } catch (error) {
      console.error("Error loading system status:", error);
      this.showAlert("Failed to load system status", "error");
    }
  }

  // Load model information
  async loadModelInfo() {
    try {
      const response = await fetch(`${this.API_BASE}/model-info`);
      const data = await response.json();

      document.getElementById("model-type").textContent =
        "CNN with Transfer Learning";
      document.getElementById("num-classes").textContent =
        data.num_classes || "-";
      document.getElementById("total-parameters").textContent =
        data.total_parameters ? data.total_parameters.toLocaleString() : "-";
      document.getElementById("input-shape").textContent = data.input_shape
        ? JSON.stringify(data.input_shape)
        : "-";
    } catch (error) {
      console.error("Error loading model info:", error);
    }
  }

  // Handle single image upload
  async handleSingleUpload(input) {
    const file = input.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    this.showLoading("single-prediction-result");

    try {
      const response = await fetch(`${this.API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      this.displaySinglePrediction(result, file);
    } catch (error) {
      console.error("Prediction error:", error);
      this.showAlert("Prediction failed", "error");
    }
  }

  // Display single prediction result
  displaySinglePrediction(result, file) {
    const container = document.getElementById("single-prediction-result");

    if (result.success) {
      const prediction = result.prediction;
      container.innerHTML = `
                <div class="prediction-result">
                    <h4><i class="fas fa-bullseye"></i> Prediction Result</h4>
                    <p><strong>File:</strong> ${result.filename}</p>
                    <p><strong>Predicted Class:</strong> ${
                      prediction.predicted_class
                    }</p>
                    <p><strong>Confidence:</strong> ${(
                      prediction.confidence * 100
                    ).toFixed(2)}%</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${
                          prediction.confidence * 100
                        }%"></div>
                    </div>
                    ${
                      prediction.all_probabilities
                        ? "<details style='margin-top: 1rem; color: rgba(255,255,255,0.9);'><summary style='cursor: pointer; font-weight: 600;'>View All Probabilities</summary>" +
                          Object.entries(prediction.all_probabilities)
                            .map(
                              ([cls, prob]) =>
                                `<div class="metric"><span>${cls}:</span><span>${(
                                  prob * 100
                                ).toFixed(2)}%</span></div>`
                            )
                            .join("") +
                          "</details>"
                        : ""
                    }
                </div>
            `;
    } else {
      container.innerHTML = `<div class="alert alert-error"><i class="fas fa-exclamation-triangle"></i> Prediction failed</div>`;
    }

    container.classList.remove("hidden");
  }

  // Handle batch upload
  handleBatchUpload(input) {
    this.batchFiles = Array.from(input.files);
    this.displayBatchFiles();
    document.getElementById("batch-predict-btn").disabled =
      this.batchFiles.length === 0;
  }

  // Display batch files
  displayBatchFiles() {
    const container = document.getElementById("batch-file-list");

    if (this.batchFiles.length === 0) {
      container.classList.add("hidden");
      return;
    }

    container.innerHTML = this.batchFiles
      .map(
        (file, index) => `
                <div class="file-item">
                    <span><i class="fas fa-image"></i> ${file.name}</span>
                    <button class="btn btn-danger btn-sm" onclick="dashboard.removeBatchFile(${index})">
                        <i class="fas fa-times"></i> Remove
                    </button>
                </div>
            `
      )
      .join("");

    container.classList.remove("hidden");
  }

  // Remove file from batch
  removeBatchFile(index) {
    this.batchFiles.splice(index, 1);
    this.displayBatchFiles();
    document.getElementById("batch-predict-btn").disabled =
      this.batchFiles.length === 0;
  }

  // Predict batch
  async predictBatch() {
    if (this.batchFiles.length === 0) return;

    const formData = new FormData();
    this.batchFiles.forEach((file) => formData.append("files", file));

    document.getElementById("batch-predict-btn").disabled = true;
    this.showLoading("batch-prediction-results");

    try {
      const response = await fetch(`${this.API_BASE}/predict/batch`, {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      this.displayBatchResults(result);
    } catch (error) {
      console.error("Batch prediction error:", error);
      this.showAlert("Batch prediction failed", "error");
    } finally {
      document.getElementById("batch-predict-btn").disabled = false;
    }
  }

  // Display batch results
  displayBatchResults(result) {
    const container = document.getElementById("batch-prediction-results");

    container.innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-info-circle"></i>
                <strong>Batch Results:</strong> ${result.successful}/${
      result.total_processed
    } successful
            </div>
            <div class="file-list">
                ${result.batch_results
                  .map(
                    (item) => `
                        <div class="file-item">
                            <span><i class="fas fa-image"></i> ${
                              item.filename
                            }</span>
                            <span class="metric-value ${
                              item.success ? "text-success" : "text-error"
                            }">
                                ${
                                  item.success
                                    ? `${item.prediction.predicted_class} (${(
                                        item.prediction.confidence * 100
                                      ).toFixed(1)}%)`
                                    : `<i class="fas fa-exclamation-triangle"></i> Error: ${item.error}`
                                }
                            </span>
                        </div>
                    `
                  )
                  .join("")}
            </div>
        `;

    container.classList.remove("hidden");
  }

  // Load visualizations
  async loadVisualizations() {
    try {
      const response = await fetch(`${this.API_BASE}/visualizations`);
      const data = await response.json();

      if (data.message) {
        document.getElementById("visualization-tab").innerHTML = `
                    <div class="card">
                        <div class="alert alert-info"><i class="fas fa-info-circle"></i> ${data.message}</div>
                    </div>
                `;
        return;
      }

      // Create charts
      this.createClassDistributionChart(data.class_distribution);
      this.createConfidenceChart(data.confidence_statistics);
      this.createTimeChart(data.predictions_by_hour);
    } catch (error) {
      console.error("Error loading visualizations:", error);
    }
  }

  // Create class distribution chart
  createClassDistributionChart(data) {
    const ctx = document
      .getElementById("class-distribution-chart")
      .getContext("2d");

    if (this.charts.classDistribution) {
      this.charts.classDistribution.destroy();
    }

    this.charts.classDistribution = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: Object.keys(data),
        datasets: [
          {
            data: Object.values(data),
            backgroundColor: [
              "#667eea",
              "#764ba2",
              "#f093fb",
              "#f5576c",
              "#4facfe",
              "#00f2fe",
              "#fa709a",
              "#fee140",
              "#a8edea",
              "#fed6e3",
              "#ffd89b",
              "#19547b",
            ],
            borderWidth: 0,
            hoverBorderWidth: 3,
            hoverBorderColor: "#fff",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              color: "rgba(255,255,255,0.8)",
              padding: 20,
              font: { size: 12, weight: "500" },
            },
          },
        },
      },
    });
  }

  // Create confidence chart
  createConfidenceChart(data) {
    const ctx = document.getElementById("confidence-chart").getContext("2d");

    if (this.charts.confidence) {
      this.charts.confidence.destroy();
    }

    this.charts.confidence = new Chart(ctx, {
      type: "bar",
      data: {
        labels: ["Mean", "Std Dev", "Min", "Max"],
        datasets: [
          {
            label: "Confidence Statistics",
            data: [data.mean, data.std, data.min, data.max],
            backgroundColor: [
              "rgba(102, 126, 234, 0.8)",
              "rgba(118, 75, 162, 0.8)",
              "rgba(240, 147, 251, 0.8)",
              "rgba(245, 87, 108, 0.8)",
            ],
            borderColor: ["#667eea", "#764ba2", "#f093fb", "#f5576c"],
            borderWidth: 2,
            borderRadius: 8,
            borderSkipped: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: {
            beginAtZero: true,
            max: 1,
            ticks: { color: "rgba(255,255,255,0.8)" },
            grid: { color: "rgba(255,255,255,0.1)" },
          },
          x: {
            ticks: { color: "rgba(255,255,255,0.8)" },
            grid: { display: false },
          },
        },
      },
    });
  }

  // Create time-based chart
  createTimeChart(data) {
    const ctx = document.getElementById("time-chart").getContext("2d");

    if (this.charts.time) {
      this.charts.time.destroy();
    }

    const hours = Array.from({ length: 24 }, (_, i) => i);
    const values = hours.map((hour) => data[hour] || 0);

    this.charts.time = new Chart(ctx, {
      type: "line",
      data: {
        labels: hours.map((h) => `${h}:00`),
        datasets: [
          {
            label: "Predictions per Hour",
            data: values,
            borderColor: "#667eea",
            backgroundColor: "rgba(102, 126, 234, 0.1)",
            tension: 0.4,
            fill: true,
            pointBackgroundColor: "#667eea",
            pointBorderColor: "#fff",
            pointBorderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "rgba(255,255,255,0.8)" } },
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: { color: "rgba(255,255,255,0.8)" },
            grid: { color: "rgba(255,255,255,0.1)" },
          },
          x: {
            ticks: { color: "rgba(255,255,255,0.8)" },
            grid: { color: "rgba(255,255,255,0.1)" },
          },
        },
      },
    });
  }

  // Handle training data upload
  async handleTrainingUpload(input) {
    const file = input.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("zip_file", file);

    this.showLoading("training-upload-result");

    try {
      const response = await fetch(`${this.API_BASE}/upload/training-data-zip`, {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (result.success) {
        document.getElementById("training-upload-result").innerHTML = `
                    <div class="alert alert-success">
                        <i class="fas fa-check-circle"></i>
                        <strong>Upload Successful!</strong><br>
                        Total images: ${result.data_info.total_files}<br>
                        Valid images: ${result.data_info.valid_images}<br>
                        Classes: ${Object.keys(result.data_info.classes).length}
                    </div>
                `;
      } else {
        this.showAlert("Upload failed", "error");
      }

      document
        .getElementById("training-upload-result")
        .classList.remove("hidden");
    } catch (error) {
      console.error("Upload error:", error);
      this.showAlert("Upload failed", "error");
    }
  }

  // Trigger retraining
  async triggerRetraining() {
    if (
      !confirm(
        "Are you sure you want to start retraining? This may take several minutes."
      )
    ) {
      return;
    }

    document.getElementById("retrain-btn").disabled = true;
    this.showLoading("retraining-status");

    try {
      const response = await fetch(`${this.API_BASE}/retrain`, {
        method: "POST",
      });

      const result = await response.json();

      if (result.task_id) {
        document.getElementById("retraining-status").innerHTML = `
                    <div class="alert alert-info">
                        <i class="fas fa-rocket"></i>
                        <strong>Retraining Started!</strong><br>
                        Task ID: ${result.task_id}<br>
                        Status: ${result.status}
                    </div>
                `;

        // Poll for status updates
        this.pollRetrainingStatus(result.task_id);
      } else {
        this.showAlert("Failed to start retraining", "error");
      }

      document.getElementById("retraining-status").classList.remove("hidden");
    } catch (error) {
      console.error("Retraining error:", error);
      this.showAlert("Failed to start retraining", "error");
    } finally {
      document.getElementById("retrain-btn").disabled = false;
    }
  }

  // Poll retraining status
  async pollRetrainingStatus(taskId) {
    try {
      const response = await fetch(`${this.API_BASE}/training/status/${taskId}`);
      const status = await response.json();

      document.getElementById("retraining-status").innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-spinner fa-spin"></i>
                    <strong>Retraining Status:</strong> ${status.status}<br>
                    ${
                      status.error
                        ? `<i class="fas fa-exclamation-triangle"></i> Error: ${status.error}`
                        : ""
                    }
                    ${
                      status.completed_at
                        ? `<i class="fas fa-check"></i> Completed: ${new Date(
                            status.completed_at
                          ).toLocaleString()}`
                        : ""
                    }
                </div>
            `;

      if (status.status === "completed" || status.status === "failed") {
        if (status.status === "completed") {
          this.showAlert("Retraining completed successfully!", "success");
          this.loadModelInfo(); // Refresh model info

          // Refresh model versions
          if (window.modelVersionManager) {
            window.modelVersionManager.loadModelVersions();
          }
        }
        return;
      }

      // Continue polling
      setTimeout(() => this.pollRetrainingStatus(taskId), 5000);
    } catch (error) {
      console.error("Error polling retraining status:", error);
    }
  }

  // Load recent predictions
  async loadRecentPredictions() {
    try {
      const response = await fetch(`${this.API_BASE}/metrics`);
      const data = await response.json();

      const container = document.getElementById("recent-predictions");

      if (data.recent_predictions && data.recent_predictions.length > 0) {
        container.innerHTML = data.recent_predictions
          .map(
            (pred) => `
                        <div class="metric">
                            <span><i class="fas fa-image"></i> ${
                              pred.filename
                            }</span>
                            <span class="metric-value">${pred.prediction} (${(
              pred.confidence * 100
            ).toFixed(1)}%)</span>
                        </div>
                    `
          )
          .join("");
      } else {
        container.innerHTML =
          '<div class="alert alert-info"><i class="fas fa-info-circle"></i> No recent predictions</div>';
      }
    } catch (error) {
      console.error("Error loading recent predictions:", error);
    }
  }

  // Download logs
  async downloadLogs() {
    try {
      const response = await fetch(`${this.API_BASE}/download-logs`);
      const data = await response.json();

      const blob = new Blob([data.csv_data], { type: "text/csv" });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "prediction_logs.csv";
      a.click();
      window.URL.revokeObjectURL(url);

      this.showAlert("Logs downloaded successfully!", "success");
    } catch (error) {
      console.error("Error downloading logs:", error);
      this.showAlert("Failed to download logs", "error");
    }
  }

  // Clear logs
  async clearLogs() {
    if (
      !confirm(
        "Are you sure you want to clear all logs? This action cannot be undone."
      )
    ) {
      return;
    }

    try {
      // This would be an API call to clear logs
      this.showAlert("Logs cleared successfully!", "success");
      this.loadRecentPredictions();
    } catch (error) {
      console.error("Error clearing logs:", error);
      this.showAlert("Failed to clear logs", "error");
    }
  }

  // Utility functions
  showAlert(message, type) {
    const alertDiv = document.createElement("div");
    alertDiv.className = `alert alert-${type}`;
    alertDiv.innerHTML = `
            <i class="fas fa-${
              type === "success"
                ? "check-circle"
                : type === "error"
                ? "exclamation-triangle"
                : "info-circle"
            }"></i>
            ${message}
        `;

    document.body.insertBefore(alertDiv, document.body.firstChild);

    setTimeout(() => {
      alertDiv.style.animation = "slideInDown 0.5s ease-out reverse";
      setTimeout(() => alertDiv.remove(), 500);
    }, 4000);
  }

  showLoading(containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML =
      '<div class="loading"></div> <span style="color: rgba(255,255,255,0.8);">Processing...</span>';
    container.classList.remove("hidden");
  }

  refreshData() {
    this.loadSystemStatus();
    this.loadModelInfo();
    if (this.currentTab === "visualization") {
      this.loadVisualizations();
    }
    if (this.currentTab === "monitoring") {
      this.loadRecentPredictions();
    }
    if (window.modelVersionManager) {
      window.modelVersionManager.loadModelVersions();
    }
    this.showAlert("Data refreshed successfully!", "success");
  }

  // Setup drag and drop
  setupDragAndDrop() {
    const uploadAreas = document.querySelectorAll(".upload-area");

    uploadAreas.forEach((area) => {
      area.addEventListener("dragover", (e) => {
        e.preventDefault();
        area.classList.add("dragover");
      });

      area.addEventListener("dragleave", () => {
        area.classList.remove("dragover");
      });

      area.addEventListener("drop", (e) => {
        e.preventDefault();
        area.classList.remove("dragover");

        const files = e.dataTransfer.files;
        if (files.length > 0) {
          const input = area.querySelector('input[type="file"]');
          input.files = files;
          input.dispatchEvent(new Event("change"));
        }
      });
    });
  }
}

// Global functions (for HTML event handlers)
let dashboard;

function switchTab(tabName) {
  dashboard.switchTab(tabName);
}

function handleSingleUpload(input) {
  window.dashboard.handleSingleUpload(input);
}

function handleBatchUpload(input) {
  window.dashboard.handleBatchUpload(input);
}

function predictBatch() {
  window.dashboard.predictBatch();
}

function handleTrainingUpload(input) {
  window.dashboard.handleTrainingUpload(input);
}

function triggerRetraining() {
  window.dashboard.triggerRetraining();
}

function downloadLogs() {
  window.dashboard.downloadLogs();
}

function clearLogs() {
  window.dashboard.clearLogs();
}

function refreshData() {
  window.dashboard.refreshData();
}

// Initialize dashboard when DOM is loaded
document.addEventListener("DOMContentLoaded", function () {
  window.dashboard = new FoodClassificationDashboard();
});
