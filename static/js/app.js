// PD Model API JavaScript Application
// Interactive features and API integration

class PDModelAPI {
    constructor() {
        this.baseURL = window.location.origin;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeTooltips();
        this.checkAPIHealth();
    }

    setupEventListeners() {
        // Form validation
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', this.handleFormSubmit.bind(this));
        });

        // Real-time input validation
        document.querySelectorAll('input[type="number"]').forEach(input => {
            input.addEventListener('input', this.validateInput.bind(this));
        });

        // Auto-calculate derived values
        this.setupAutoCalculations();
    }

    setupAutoCalculations() {
        // Debt-to-Income helper
        const incomeInput = document.getElementById('income');
        const dtiInput = document.getElementById('debt_to_income');
        
        if (incomeInput && dtiInput) {
            [incomeInput, dtiInput].forEach(input => {
                input.addEventListener('input', () => {
                    this.updateDebtCalculation(incomeInput, dtiInput);
                });
            });
        }

        // Credit utilization helper
        const utilizationInput = document.getElementById('utilization_rate');
        if (utilizationInput) {
            utilizationInput.addEventListener('input', (e) => {
                this.updateUtilizationHelper(e.target);
            });
        }
    }

    updateDebtCalculation(incomeInput, dtiInput) {
        const income = parseFloat(incomeInput.value) || 0;
        const dti = parseFloat(dtiInput.value) || 0;
        const totalDebt = income * dti;
        
        if (income > 0 && dti > 0) {
            const helper = dtiInput.parentElement.querySelector('.debt-helper');
            if (!helper) {
                const helperDiv = document.createElement('div');
                helperDiv.className = 'debt-helper mt-1 text-info small';
                dtiInput.parentElement.appendChild(helperDiv);
            }
            
            const helperElement = dtiInput.parentElement.querySelector('.debt-helper');
            helperElement.innerHTML = `
                <i class="fas fa-calculator me-1"></i>
                Estimated total debt: $${totalDebt.toLocaleString()}
            `;
        }
    }

    updateUtilizationHelper(input) {
        const utilization = parseFloat(input.value) || 0;
        const percentage = (utilization * 100).toFixed(1);
        
        let helperElement = input.parentElement.querySelector('.utilization-helper');
        if (!helperElement) {
            helperElement = document.createElement('div');
            helperElement.className = 'utilization-helper mt-1 small';
            input.parentElement.appendChild(helperElement);
        }
        
        let colorClass = 'text-success';
        let icon = 'fas fa-check-circle';
        let message = 'Excellent utilization';
        
        if (utilization > 0.3) {
            colorClass = 'text-warning';
            icon = 'fas fa-exclamation-triangle';
            message = 'Moderate utilization';
        }
        if (utilization > 0.7) {
            colorClass = 'text-danger';
            icon = 'fas fa-times-circle';
            message = 'High utilization - may impact score';
        }
        
        helperElement.className = `utilization-helper mt-1 small ${colorClass}`;
        helperElement.innerHTML = `
            <i class="${icon} me-1"></i>
            ${percentage}% - ${message}
        `;
    }

    validateInput(event) {
        const input = event.target;
        const value = parseFloat(input.value);
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);
        
        input.classList.remove('is-valid', 'is-invalid');
        
        if (isNaN(value)) {
            this.showInputFeedback(input, 'Please enter a valid number', false);
        } else if (min !== undefined && value < min) {
            this.showInputFeedback(input, `Value must be at least ${min}`, false);
        } else if (max !== undefined && value > max) {
            this.showInputFeedback(input, `Value must be at most ${max}`, false);
        } else {
            this.showInputFeedback(input, 'Valid input', true);
        }
    }

    showInputFeedback(input, message, isValid) {
        input.classList.add(isValid ? 'is-valid' : 'is-invalid');
        
        // Remove existing feedback
        const existingFeedback = input.parentElement.querySelector('.invalid-feedback, .valid-feedback');
        if (existingFeedback) {
            existingFeedback.remove();
        }
        
        // Add new feedback
        const feedbackDiv = document.createElement('div');
        feedbackDiv.className = isValid ? 'valid-feedback' : 'invalid-feedback';
        feedbackDiv.textContent = message;
        input.parentElement.appendChild(feedbackDiv);
    }

    handleFormSubmit(event) {
        const form = event.target;
        const submitBtn = form.querySelector('button[type="submit"]');
        
        if (submitBtn) {
            const originalText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Calculating...';
            submitBtn.disabled = true;
            
            // Re-enable button if form submission fails
            setTimeout(() => {
                if (submitBtn.disabled) {
                    submitBtn.innerHTML = originalText;
                    submitBtn.disabled = false;
                }
            }, 30000); // 30 second timeout
        }
    }

    initializeTooltips() {
        // Initialize Bootstrap tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    async checkAPIHealth() {
        try {
            const response = await fetch(`${this.baseURL}/api/health`);
            const data = await response.json();
            
            if (data.status === 'healthy') {
                this.updateHealthStatus(true, data);
            } else {
                this.updateHealthStatus(false, 'API is not healthy');
            }
        } catch (error) {
            this.updateHealthStatus(false, 'API is unavailable');
        }
    }

    updateHealthStatus(isHealthy, data) {
        const statusElements = document.querySelectorAll('.api-status');
        statusElements.forEach(element => {
            if (isHealthy) {
                element.innerHTML = '<i class="fas fa-check-circle text-success me-1"></i>API Online';
                element.className = 'api-status badge bg-success';
            } else {
                element.innerHTML = '<i class="fas fa-times-circle text-danger me-1"></i>API Offline';
                element.className = 'api-status badge bg-danger';
            }
        });

        // Update model status if available
        if (isHealthy && data.models) {
            this.updateModelStatus(data.models);
        }
    }

    updateModelStatus(models) {
        Object.keys(models).forEach(segment => {
            const statusElement = document.querySelector(`#${segment}-status`);
            if (statusElement) {
                const isLoaded = models[segment].loaded;
                const modelCount = models[segment].model_count;
                
                statusElement.innerHTML = isLoaded 
                    ? `<i class="fas fa-check-circle text-success me-1"></i>${modelCount} models loaded`
                    : `<i class="fas fa-times-circle text-danger me-1"></i>Models not loaded`;
                    
                statusElement.className = isLoaded 
                    ? 'badge bg-success' 
                    : 'badge bg-danger';
            }
        });
    }

    // Utility methods
    formatCurrency(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(value);
    }

    formatPercentage(value, decimals = 2) {
        return (value * 100).toFixed(decimals) + '%';
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.top = '20px';
        notification.style.right = '20px';
        notification.style.zIndex = '9999';
        notification.style.minWidth = '300px';
        
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }

    // API methods for AJAX calls
    async predict(segment, data) {
        try {
            const response = await fetch(`${this.baseURL}/api/predict/${segment}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Prediction error:', error);
            this.showNotification('Prediction failed. Please try again.', 'danger');
            throw error;
        }
    }

    async batchPredict(segment, file) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('segment', segment);
            
            const response = await fetch(`${this.baseURL}/api/predict/batch`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Batch prediction error:', error);
            this.showNotification('Batch prediction failed. Please try again.', 'danger');
            throw error;
        }
    }
}

// Interactive features for specific pages
class ResultsPage {
    constructor() {
        this.initializeAnimations();
        this.setupExportFeatures();
    }

    initializeAnimations() {
        // Animate result cards on load
        const resultCards = document.querySelectorAll('.result-card');
        resultCards.forEach((card, index) => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(30px)';
            
            setTimeout(() => {
                card.style.transition = 'all 0.6s ease';
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, index * 200);
        });

        // Animate progress bars
        setTimeout(() => {
            const progressBars = document.querySelectorAll('.progress-bar');
            progressBars.forEach(bar => {
                const width = bar.style.width;
                bar.style.width = '0%';
                setTimeout(() => {
                    bar.style.width = width;
                }, 500);
            });
        }, 1000);
    }

    setupExportFeatures() {
        // CSV export
        window.exportCSV = () => {
            const data = this.extractResultData();
            const csv = this.convertToCSV(data);
            this.downloadFile(csv, 'pd_results.csv', 'text/csv');
        };

        // JSON export
        window.exportJSON = () => {
            const data = this.extractResultData();
            const json = JSON.stringify(data, null, 2);
            this.downloadFile(json, 'pd_results.json', 'application/json');
        };
    }

    extractResultData() {
        return {
            pd_score: parseFloat(document.querySelector('.result-value').textContent),
            risk_grade: document.querySelectorAll('.result-value')[1].textContent,
            ifrs9_stage: parseInt(document.querySelectorAll('.result-value')[2].textContent.replace('Stage ', '')),
            timestamp: new Date().toISOString(),
            segment: window.location.pathname.split('/')[1] || 'unknown'
        };
    }

    convertToCSV(data) {
        const headers = Object.keys(data).join(',');
        const values = Object.values(data).join(',');
        return `${headers}\n${values}`;
    }

    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
}

// Batch processing features
class BatchProcessor {
    constructor() {
        this.setupFileUpload();
        this.setupProgress();
    }

    setupFileUpload() {
        const fileInput = document.getElementById('batch-file');
        const uploadArea = document.getElementById('upload-area');
        
        if (fileInput && uploadArea) {
            // Drag and drop
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('drag-over');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('drag-over');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('drag-over');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    fileInput.files = files;
                    this.handleFileSelect(files[0]);
                }
            });
            
            // Click to upload
            uploadArea.addEventListener('click', () => {
                fileInput.click();
            });
            
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.handleFileSelect(e.target.files[0]);
                }
            });
        }
    }

    handleFileSelect(file) {
        const fileName = document.getElementById('file-name');
        const fileSize = document.getElementById('file-size');
        
        if (fileName) fileName.textContent = file.name;
        if (fileSize) fileSize.textContent = this.formatFileSize(file.size);
        
        // Validate file
        if (!file.name.endsWith('.csv')) {
            pdAPI.showNotification('Please select a CSV file', 'warning');
            return;
        }
        
        if (file.size > 10 * 1024 * 1024) { // 10MB limit
            pdAPI.showNotification('File size must be less than 10MB', 'warning');
            return;
        }
        
        // Show preview if possible
        this.previewFile(file);
    }

    previewFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const csv = e.target.result;
            const lines = csv.split('\n').slice(0, 6); // First 5 rows + header
            const preview = document.getElementById('file-preview');
            
            if (preview) {
                preview.innerHTML = `
                    <h6>File Preview (first 5 rows):</h6>
                    <pre class="bg-light p-2 rounded small">${lines.join('\n')}</pre>
                `;
            }
        };
        reader.readAsText(file);
    }

    setupProgress() {
        // Progress tracking for batch operations
        this.progressBar = document.getElementById('batch-progress');
        this.progressText = document.getElementById('progress-text');
    }

    updateProgress(percentage, text = '') {
        if (this.progressBar) {
            this.progressBar.style.width = `${percentage}%`;
            this.progressBar.setAttribute('aria-valuenow', percentage);
        }
        
        if (this.progressText && text) {
            this.progressText.textContent = text;
        }
    }

    formatFileSize(bytes) {
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        if (bytes === 0) return '0 Bytes';
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize main API class
    window.pdAPI = new PDModelAPI();
    
    // Initialize page-specific features
    if (document.querySelector('.result-card')) {
        new ResultsPage();
    }
    
    if (document.getElementById('batch-file')) {
        new BatchProcessor();
    }
    
    // Global utility functions
    window.resetForm = (formId) => {
        const form = document.getElementById(formId) || document.querySelector('form');
        if (form) {
            form.reset();
            form.querySelectorAll('.is-valid, .is-invalid').forEach(el => {
                el.classList.remove('is-valid', 'is-invalid');
            });
            form.querySelectorAll('.invalid-feedback, .valid-feedback').forEach(el => {
                el.remove();
            });
        }
    };
    
    // Global keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl+Enter to submit forms
        if (e.ctrlKey && e.key === 'Enter') {
            const activeForm = document.querySelector('form:focus-within');
            if (activeForm) {
                activeForm.submit();
            }
        }
    });
    
    console.log('🏦 PD Model API JavaScript initialized');
});