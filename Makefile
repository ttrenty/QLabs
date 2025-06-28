# Makefile that forward every target to `pixi run`

.PHONY: all
all:
	# @pixi run test
	@pixi run main
	@pixi run bench

.PHONY: %
%:
	@pixi run $@ $(ARGS)