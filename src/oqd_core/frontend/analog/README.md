# Frontend for Analog Interface

Make sure you're in the virtual environment, and sync your requirements.

```bash
uv sync
source .venv/bin/activate
```

# Convert .analog into the AST
The example code is in `examples/analog/test.analog`. This file contains the expected syntax of the Parser.

Run the Typer app:
```bash
# Run example
python3 -m oqd_core.frontend.analog.output -i examples/analog/test.analog -o examples/analog/test.ast
```

This example generates the Analog Circuit AST in `examples/analog/test.ast`.

# Deserialize JSON AST into .analog code

Run the Typer app with the deserialize option:
```bash
# Run example
python3 -m oqd_core.frontend.analog.output -d -i examples/analog/test.ast -o examples/analog/test.out
```
This example generates the Analog language code in `examples/analog/test.out`.