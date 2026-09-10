PYTHON ?= python

.PHONY: install install-core install-common install-tabletop install-excavator install-excavator-operations install-workbench install-operations test

install: install-tabletop

install-core:
	$(PYTHON) -m pip install -e ../stacking-core/python

install-common: install-core
	$(PYTHON) -m pip install -e packages/stacking-common

install-tabletop: install-common
	$(PYTHON) -m pip install -e 'apps/tabletop[perception]'

install-excavator: install-common
	$(PYTHON) -m pip install -e apps/excavator

install-excavator-operations: install-common
	$(PYTHON) -m pip install -e 'apps/excavator[operations]'

install-workbench: install-common
	$(PYTHON) -m pip install -e 'apps/tabletop[perception,workbench]'

install-operations: install-common
	$(PYTHON) -m pip install -e 'apps/tabletop[operations]'

test:
	$(PYTHON) -m pytest -q
