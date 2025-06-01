// PD Model API JavaScript Application
// Enhanced with better error handling and mobile support

class PDModelAPI {
    constructor() {
        this.baseURL = window.location.origin;
        this.retryAttempts = 3;
        this.retryDelay = 1000;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeComponents();
        this.setupErrorHandling();
        this.checkAPIHealth();
        this.setupFormValidation();
    }

    setupEventListeners() {
        // Form submission handling
        document.addEventListener('submit', this.handleFormSubmit.bind(this));
        
        // Real-time input validation
        document.addEventListener('input', this.handleInputChange.bind(this));
        
        // Auto-calculation setup
        this.setupAutoCalculations();
        
        // Mobile-specific interactions
        this.setupMobileInteractions();
        
        // Keyboard shortcuts
        this.setupKeyboardShortcuts();
    }

    initializeComponents() {
        // Initialize tooltips
        this.initializeTooltips();
        
        // Initialize file upload
        this.initializeFileUpload();
        
        // Initialize progress tracking
        this.initializeProgressTracking();
        
        // Initialize notifications
        this.initializeNotifications();
    }

    setupErrorHandling() {
        // Global error handler
        window.addEventListener('error', (event) => {
            console.error('Global error:', event.error);
            this.logError('Global Error', event.error);
        });

        // Unhandled promise rejection handler
        window.addEventListener('unhandledrejection', (event) => {
            console.error('Unhandled promise rejection:', event.reason);
            this.logError('Promise Rejection', event.reason);
        });

        // Network error detection
        window.addEventListener('online', () => {
            this.showNotification('Connection restored', 'success');
        });

        window.addEventListener('offline', () => {
            this.showNotification('Connection lost. Please check your internet connection.', 'warning');
        });
    }

    setupFormValidation() {
        // Real-time form validation
        const forms = document.querySelectorAll('form');
        forms.forEach(form => {
            this.addFormValidation(form);
        });
    }

    setupAutoCalculations() {
        // Debt-to-Income helper
        this.setupDTICalculation();
        
        // Credit utilization helper
        this.setupUtilizationHelper();
        
        // Revenue per employee calculation
        this.setupRevenuePerEmployeeHelper();
        
        // Financial ratio helpers
        this.setupFinancialRatioHelpers();
    }

    setupDTICalculation() {
        const incomeInput = document.getElementById('income');
        const dtiInput = document.getElementById('debt_to_income');
        
        if (incomeInput && dtiInput) {
            [incomeInput, dtiInput].forEach(input => {
                input.addEventListener('input', this.debounce(() => {
                    this.updateDebtCalculation(incomeInput, dtiInput);
                }, 300));
            });
        }
    }

    setupUtilizationHelper() {
        const utilizationInput = document.getElementById('utilization_rate');
        if (utilizationInput) {
            utilizationInput.addEventListener('input', this.debounce((e) => {
                this.updateUtilizationHelper(e.target);
            }, 300));
        }
    }

    setupRevenuePerEmployeeHelper() {
        const revenueInput = document.getElementById('annual_revenue');
        const employeeInput = document.getElementById('num_employees');
        
        if (revenueInput && employeeInput) {
            [revenueInput, employeeInput].forEach(input => {
                input.addEventListener('input', this.debounce(() => {
                    this.updateRevenuePerEmployee(revenueInput, employeeInput);
                }, 300));
            });
        }
    }

    setupFinancialRatioHelpers() {
        // Current ratio helper
        const currentRatioInput = document.getElementById('current_ratio');
        if (currentRatioInput) {
            currentRatioInput.addEventListener('input', this.debounce((e) => {
                this.updateRatioHelper(e.target, 'current-ratio');
            }, 300));
        }

        // Debt-to-equity helper
        const debtEquityInput = document.getElementById('debt_to_equity');
        if (debtEquityInput) {
            debtEquityInput.addEventListener('input', this.debounce((e) => {
                this.updateRatioHelper(e.target, 'debt-equity');
            }, 300));
        }

        // Profit margin helper
        const profitMarginInput = document.getElementById('profit_margin');
        if (profitMarginInput) {
            profitMarginInput.addEventListener('input', this.debounce((e) => {
                this.updateRatioHelper(e.target, 'profit-margin');
            }, 300));
        }
    }

    setupMobileInteractions() {
        // Touch-friendly interactions
        this.setupTouchGestures();
        
        // Mobile form optimizations
        this.setupMobileFormOptimizations();
        
        // Responsive navigation
        this.setupResponsiveNavigation();
    }

    setupTouchGestures() {
        // Add touch-friendly interactions for mobile
        const cards = document.querySelectorAll('.segment-card, .result-card');
        cards.forEach(card => {
            card.addEventListener('touchstart', function() {
                this.style.transform = 'scale(0.98)';
            });
            
            card.addEventListener('touchend', function() {
                this.style.transform = '';
            });
        });
    }

    setupMobileFormOptimizations() {
        // Auto-focus management for mobile
        const firstInput = document.querySelector('form input:not([type="hidden"])');
        if (firstInput && window.innerWidth > 768) {
            // Only auto-focus on desktop to avoid mobile keyboard issues
            firstInput.focus();
        }

        // Improve number input experience on mobile
        const numberInputs = document.querySelectorAll('input[type="number"]');
        numberInputs.forEach(input => {
            input.addEventListener('focus', function() {
                // Scroll into view on mobile
                if (window.innerWidth <= 768) {
                    setTimeout(() => {
                        this.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }, 300);
                }
            });
        });
    }

    setupResponsiveNavigation() {
        const navbarToggler = document.querySelector('.navbar-toggler');
        const navbarCollapse = document.querySelector('.navbar-collapse');
        
        if (navbarToggler && navbarCollapse) {
            // Close menu when clicking outside
            document.addEventListener('click', (e) => {
                if (!navbarToggler.contains(e.target) && 
                    !navbarCollapse.contains(e.target) && 
                    navbarCollapse.classList.contains('show')) {
                    
                    const bsCollapse = new bootstrap.Collapse(navbarCollapse, {
                        toggle: false
                    });
                    bsCollapse.hide();
                }
            });
        }
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + Enter to submit forms
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                const activeForm = document.querySelector('form:focus-within');
                if (activeForm) {
                    e.preventDefault();
                    activeForm.requestSubmit();
                }
            }
            
            // Escape to close modals/menus
            if (e.key === 'Escape') {
                const openModal = document.querySelector('.modal.show');
                if (openModal) {
                    const modal = bootstrap.Modal.getInstance(openModal);
                    if (modal) modal.hide();
                }
            }
        });
    }

    handleFormSubmit(event) {
        const form = event.target;
        if (!form.matches('form')) return;
        
        const submitBtn = form.querySelector('button[type="submit"]');
        
        if (submitBtn && !submitBtn.disabled) {
            const originalText = submitBtn.innerHTML;
            const loadingText = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            
            submitBtn.innerHTML = loadingText;
            submitBtn.disabled = true;
            
            // Add timeout to re-enable button if submission fails
            const timeout = setTimeout(() => {
                if (submitBtn.disabled) {
                    submitBtn.innerHTML = originalText;
                    submitBtn.disabled = false;
                    this.showNotification('Request timed out. Please try again.', 'warning');
                }
            }, 30000);
            
            // Store timeout ID to clear it if form succeeds
            form.dataset.timeoutId = timeout;
        }
        
        // Validate form before submission
        if (!this.validateForm(form)) {
            event.preventDefault();
            if (submitBtn) {
                submitBtn.innerHTML = submitBtn.dataset.originalText || 'Submit';
                submitBtn.disabled = false;
            }
        }
    }

    handleInputChange(event) {
        const input = event.target;
        if (!input.matches('input, select, textarea')) return;
        
        // Real-time validation
        this.validateInput(input);
        
        // Update dependent calculations
        this.updateCalculations(input);
    }

    validateForm(form) {
        const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');
        let isValid = true;
        
        inputs.forEach(input => {
            if (!this.validateInput(input)) {
                isValid = false;
            }
        });
        
        return isValid;
    }

    validateInput(input) {
        const value = input.value.trim();
        const type = input.type;
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);
        let isValid = true;
        let message = '';
        
        // Clear existing validation
        input.classList.remove('is-valid', 'is-invalid');
        this.removeValidationFeedback(input);
        
        // Required field validation
        if (input.hasAttribute('required') && !value) {
            isValid = false;
            message = 'This field is required';
        }
        
        // Type-specific validation
        if (value && type === 'number') {
            const numValue = parseFloat(value);
            
            if (isNaN(numValue)) {
                isValid = false;
                message = 'Please enter a valid number';
            } else if (!isNaN(min) && numValue < min) {
                isValid = false;
                message = `Value must be at least ${min}`;
            } else if (!isNaN(max) && numValue > max) {
                isValid = false;
                message = `Value must be at most ${max}`;
            }
        }
        
        // Email validation
        if (value && type === 'email') {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(value)) {
                isValid = false;
                message = 'Please enter a valid email address';
            }
        }
        
        // Custom business logic validation
        if (value && input.id) {
            const customValidation = this.getCustomValidation(input.id, value);
            if (!customValidation.isValid) {
                isValid = false;
                message = customValidation.message;
            }
        }
        
        // Show validation result
        if (isValid) {
            input.classList.add('is-valid');
            this.showValidationFeedback(input, 'Valid input', true);
        } else {
            input.classList.add('is-invalid');
            this.showValidationFeedback(input, message, false);
        }
        
        return isValid;
    }

    getCustomValidation(inputId, value) {
        const numValue = parseFloat(value);
        
        switch (inputId) {
            case 'utilization_rate':
                if (numValue > 0.9) {
                    return { isValid: false, message: 'Utilization rate above 90% is very high risk' };
                }
                break;
                
            case 'debt_to_income':
                if (numValue > 1.0) {
                    return { isValid: false, message: 'Debt-to-income ratio above 100% is unsustainable' };
                }
                break;
                
            case 'current_ratio':
                if (numValue < 0.5) {
                    return { isValid: false, message: 'Current ratio below 0.5 indicates liquidity issues' };
                }
                break;
                
            case 'profit_margin':
                if (numValue < -0.5) {
                    return { isValid: false, message: 'Profit margin below -50% indicates severe losses' };
                }
                break;
        }
        
        return { isValid: true };
    }

    updateDebtCalculation(incomeInput, dtiInput) {
        const income = parseFloat(incomeInput.value) || 0;
        const dti = parseFloat(dtiInput.value) || 0;
        const totalDebt = income * dti;
        
        if (income > 0 && dti > 0) {
            this.updateHelper(dtiInput, 'debt-helper', 
                `<i class="fas fa-calculator me-1"></i>Estimated total debt: ${this.formatCurrency(totalDebt)}`
            );
        } else {
            this.removeHelper(dtiInput, 'debt-helper');
        }
    }

    updateUtilizationHelper(input) {
        const utilization = parseFloat(input.value) || 0;
        const percentage = (utilization * 100).toFixed(1);
        
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
        
        this.updateHelper(input, 'utilization-helper', 
            `<i class="${icon} me-1"></i>${percentage}% - ${message}`, 
            colorClass
        );
    }

    updateRevenuePerEmployee(revenueInput, employeeInput) {
        const revenue = parseFloat(revenueInput.value) || 0;
        const employees = parseFloat(employeeInput.value) || 0;
        
        if (revenue > 0 && employees > 0) {
            const revenuePerEmployee = revenue / employees;
            this.updateHelper(employeeInput, 'revenue-helper',
                `<i class="fas fa-calculator me-1"></i>Revenue per employee: ${this.formatCurrency(revenuePerEmployee)}`
            );
        } else {
            this.removeHelper(employeeInput, 'revenue-helper');
        }
    }

    updateRatioHelper(input, type) {
        const value = parseFloat(input.value) || 0;
        let message = '';
        let colorClass = 'text-info';
        let icon = 'fas fa-info-circle';
        
        switch (type) {
            case 'current-ratio':
                if (value < 1.0) {
                    colorClass = 'text-danger';
                    icon = 'fas fa-exclamation-triangle';
                    message = 'Low liquidity - potential cash flow issues';
                } else if (value > 3.0) {
                    colorClass = 'text-warning';
                    icon = 'fas fa-info-circle';
                    message = 'High liquidity - may indicate inefficient asset use';
                } else {
                    colorClass = 'text-success';
                    icon = 'fas fa-check-circle';
                    message = 'Healthy liquidity position';
                }
                break;
                
            case 'debt-equity':
                if (value > 2.0) {
                    colorClass = 'text-danger';
                    icon = 'fas fa-exclamation-triangle';
                    message = 'High leverage - increased financial risk';
                } else if (value > 1.0) {
                    colorClass = 'text-warning';
                    icon = 'fas fa-info-circle';
                    message = 'Moderate leverage level';
                } else {
                    colorClass = 'text-success';
                    icon = 'fas fa-check-circle';
                    message = 'Conservative leverage';
                }
                break;
                
            case 'profit-margin':
                const percentage = (value * 100).toFixed(1);
                if (value < 0) {
                    colorClass = 'text-danger';
                    icon = 'fas fa-trending-down';
                    message = `${percentage}% - Company is operating at a loss`;
                } else if (value < 0.05) {
                    colorClass = 'text-warning';
                    icon = 'fas fa-minus-circle';
                    message = `${percentage}% - Low profitability`;
                } else {
                    colorClass = 'text-success';
                    icon = 'fas fa-trending-up';
                    message = `${percentage}% - Healthy profitability`;
                }
                break;
        }
        
        if (message) {
            this.updateHelper(input, `${type}-helper`, 
                `<i class="${icon} me-1"></i>${message}`, 
                colorClass
            );
        }
    }

    updateHelper(input, helperClass, content, colorClass = 'text-info') {
        let helper = input.parentElement.querySelector(`.${helperClass}`);
        if (!helper) {
            helper = document.createElement('div');
            helper.className = `${helperClass} mt-1 small ${colorClass}`;
            input.parentElement.appendChild(helper);
        }
        
        helper.className = `${helperClass} mt-1 small ${colorClass}`;
        helper.innerHTML = content;
    }

    removeHelper(input, helperClass) {
        const helper = input.parentElement.querySelector(`.${helperClass}`);
        if (helper) {
            helper.remove();
        }
    }

    showValidationFeedback(input, message, isValid) {
        this.removeValidationFeedback(input);
        
        const feedback = document.createElement('div');
        feedback.className = isValid ? 'valid-feedback' : 'invalid-feedback';
        feedback.textContent = message;
        input.parentElement.appendChild(feedback);
    }

    removeValidationFeedback(input) {
        const existingFeedback = input.parentElement.querySelectorAll('.invalid-feedback, .valid-feedback');
        existingFeedback.forEach(el => el.remove());
    }

    initializeTooltips() {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl, {
                delay: { show: 500, hide: 100 }
            });
        });
    }

    initializeFileUpload() {
        const fileInput = document.getElementById('batch-file');
        const uploadArea = document.getElementById('upload-area');
        
        if (fileInput && uploadArea) {
            this.setupDragAndDrop(fileInput, uploadArea);
            this.setupFileValidation(fileInput);
        }
    }

    setupDragAndDrop(fileInput, uploadArea) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.add('drag-over'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('drag-over'), false);
        });

        uploadArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0], fileInput);
            }
        });

        uploadArea.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0], fileInput);
            }
        });
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    handleFileSelect(file, fileInput) {
        if (!this.validateFile(file)) return;
        
        // Update file input
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;
        
        // Update UI
        this.updateFileInfo(file);
        this.previewFile(file);
        this.enableSubmitButton();
    }

    validateFile(file) {
        const maxSize = 10 * 1024 * 1024; // 10MB
        const allowedTypes = ['.csv', 'text/csv', 'application/csv'];
        
        if (file.size > maxSize) {
            this.showNotification('File size must be less than 10MB', 'warning');
            return false;
        }
        
        const isValidType = allowedTypes.some(type => 
            file.name.toLowerCase().endsWith('.csv') || file.type === type
        );
        
        if (!isValidType) {
            this.showNotification('Please select a CSV file', 'warning');
            return false;
        }
        
        return true;
    }

    updateFileInfo(file) {
        const fileName = document.getElementById('file-name');
        const fileSize = document.getElementById('file-size');
        const fileInfo = document.getElementById('file-info');
        
        if (fileName) fileName.textContent = file.name;
        if (fileSize) fileSize.textContent = this.formatFileSize(file.size);
        if (fileInfo) fileInfo.style.display = 'block';
    }

    previewFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const csv = e.target.result;
            const lines = csv.split('\n').slice(0, 6); // First 5 rows + header
            const preview = document.getElementById('file-preview');
            const previewContent = document.getElementById('preview-content');
            
            if (preview && previewContent) {
                previewContent.textContent = lines.join('\n');
                preview.style.display = 'block';
            }
        };
        reader.readAsText(file);
    }

    enableSubmitButton() {
        const submitBtn = document.getElementById('submitBtn');
        const segment = document.getElementById('segment')?.value;
        
        if (submitBtn && segment) {
            submitBtn.disabled = false;
        }
    }

    async checkAPIHealth() {
        try {
            const response = await this.fetchWithRetry('/api/health');
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
                element.innerHTML = '<i class="fas fa-check-circle me-1"></i>API Online';
                element.className = 'api-status badge bg-success';
            } else {
                element.innerHTML = '<i class="fas fa-times-circle me-1"></i>API Offline';
                element.className = 'api-status badge bg-danger';
            }
        });
    }

    // Utility methods
    async fetchWithRetry(url, options = {}, attempts = this.retryAttempts) {
        try {
            const response = await fetch(url, {
                ...options,
                timeout: 30000, // 30 second timeout
            });
            
            if (!response.ok && attempts > 1) {
                await this.delay(this.retryDelay);
                return this.fetchWithRetry(url, options, attempts - 1);
            }
            
            return response;
        } catch (error) {
            if (attempts > 1) {
                await this.delay(this.retryDelay);
                return this.fetchWithRetry(url, options, attempts - 1);
            }
            throw error;
        }
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    formatCurrency(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 0,
            maximumFractionDigits: 0
        }).format(value);
    }

    formatFileSize(bytes) {
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        if (bytes === 0) return '0 Bytes';
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    showNotification(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = `
            top: 20px;
            right: 20px;
            z-index: 9999;
            min-width: 300px;
            max-width: 400px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        `;
        
        notification.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="fas fa-${this.getNotificationIcon(type)} me-2"></i>
                <div class="flex-grow-1">${message}</div>
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after duration
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, duration);
        
        // Add click-to-dismiss
        notification.addEventListener('click', () => {
            if (notification.parentNode) {
                notification.remove();
            }
        });
    }

    getNotificationIcon(type) {
        switch (type) {
            case 'success': return 'check-circle';
            case 'warning': return 'exclamation-triangle';
            case 'danger': return 'times-circle';
            case 'info':
            default: return 'info-circle';
        }
    }

    logError(context, error) {
        const errorData = {
            context,
            message: error.message || error,
            stack: error.stack,
            timestamp: new Date().toISOString(),
            url: window.location.href,
            userAgent: navigator.userAgent
        };
        
        // Log to console for development
        console.error('Error logged:', errorData);
        
        // In production, you might want to send this to an error tracking service
        // this.sendErrorToService(errorData);
    }

    // Global utility functions
    static resetForm(formId) {
        const form = document.getElementById(formId) || document.querySelector('form');
        if (form) {
            form.reset();
            
            // Clear validation states
            form.querySelectorAll('.is-valid, .is-invalid').forEach(el => {
                el.classList.remove('is-valid', 'is-invalid');
            });
            
            // Remove feedback elements
            form.querySelectorAll('.invalid-feedback, .valid-feedback').forEach(el => {
                el.remove();
            });
            
            // Remove helpers
            form.querySelectorAll('[class*="-helper"]').forEach(el => {
                el.remove();
            });
            
            // Reset file upload UI
            const fileInfo = document.getElementById('file-info');
            const filePreview = document.getElementById('file-preview');
            if (fileInfo) fileInfo.style.display = 'none';
            if (filePreview) filePreview.style.display = 'none';
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize main API class
    window.pdAPI = new PDModelAPI();
    
    // Make resetForm available globally
    window.resetForm = PDModelAPI.resetForm;
    
    console.log('🏦 PD Model API JavaScript initialized');
});