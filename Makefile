.PHONY: format lint clean

# 기본 Python 파일 경로
PYTHON_FILES := image_classification

# 가상환경 확인
VENV_NAME := .venv
VENV_BIN := $(VENV_NAME)/bin

# 포맷팅 도구
BLACK := black
ISORT := isort
AUTOFLAKE := autoflake

# 린트 도구
FLAKE8 := flake8
MYPY := mypy

format:
	@echo "🎨 Formatting Python code..."
	$(ISORT) $(PYTHON_FILES)
	$(BLACK) $(PYTHON_FILES)
	$(AUTOFLAKE) --recursive --in-place --remove-all-unused-imports $(PYTHON_FILES)
	@echo "✨ Formatting complete!"

lint:
	@echo "🔍 Checking code style..."
	$(FLAKE8) $(PYTHON_FILES)
	$(MYPY) $(PYTHON_FILES)
	@echo "✅ Code check complete!"

clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@echo "✨ Clean complete!" 