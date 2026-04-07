# Frontend for Atomic Interface

Make sure you're in the virtual environment, and sync your requirements.

```bash
uv sync
source .venv/bin/activate
```

# Convert .atomic into the AST
The example code is in `examples/atomic/test.atomic`. This file contains the expected syntax of the Parser.

Run the Typer app:
```bash
# Run example
python3 -m oqd_core.frontend.atomic.output -i examples/atomic/test.atomic -o examples/atomic/test.ast
```

This example generates the Atomic Circuit AST in `examples/atomic/test.ast`.

# Deserialize JSON AST into .atomic code

Run the Typer app with the deserialize option:
```bash
# Run example
python3 -m oqd_core.frontend.atomic.output -d -i examples/atomic/test.ast -o examples/atomic/test.out
```
This example generates the Atomic language code in `examples/atomic/test.out`.