# Makefile that forward every target to `pixi run`

.PHONY: all
all:
	@pixi run main
	@pixi run test
	# @pixi run bench

.PHONY: %
%:
	@pixi run $@ $(ARGS)