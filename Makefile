# Makefile that forward every target to `pixi run`
.PHONY: %
%:
	@pixi run $@ $(ARGS)