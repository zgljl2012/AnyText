


clear:
	@rm -rf dist build

clear-win:
	@rd /s /q "build"
	@rd /s /q "dist"

build:
	@python setup.py bdist_wheel

build-win: clear-win build

publish: build
	@python -m twine upload dist/*

publish-win: build-win
	@python -m twine upload dist/*
