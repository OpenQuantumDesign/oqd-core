# Copyright 2024-2025 Open Quantum Design

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .AtomicLexer import AtomicLexer
from .AtomicParser import AtomicParser
# from .AtomicCircuitAST import _AtomicASTBuilder
from pathlib import Path
import antlr4
import typer

# from oqd_core.compiler.analog.passes.canonicalize import analog_operator_canonicalization

########################################################################################

app = typer.Typer()

def to_string_tree(root, symbolic_lexer_names, token_delimiter='"', show_token_types=True):
    builder = []
    _to_string_tree_traverse(root, builder, symbolic_lexer_names, token_delimiter, show_token_types)
    return ''.join(builder)


def _to_string_tree_traverse(tree, builder, symbolic_lexer_names, token_delimiter, show_token_types):
    child_list_stack = [[tree]]

    while len(child_list_stack) > 0:
        child_stack = child_list_stack[-1]

        if len(child_stack) == 0:
            child_list_stack.pop()
        else:
            tree = child_stack.pop(0)
            node = str(type(tree).__name__).replace('Context', '')
            node = '{0}{1}'.format(node[0].lower(), node[1:])

            indent = []

            for i in range(0, len(child_list_stack) - 1):
                indent.append('│  ' if len(child_list_stack[i]) > 0 else '   ')

            token_name = ''
            token_type = tree.getPayload().type if node.startswith('terminal') else 0

            if show_token_types and token_type > -1:
                token_name = ' ({0})'.format(symbolic_lexer_names[token_type])

            builder.extend(indent)
            builder.append('└── ' if len(child_stack) == 0 else '├── ')
            builder.append('{0}{1}{2}{3}'.format(token_delimiter, tree.getText(), token_delimiter, token_name)
                           if node.startswith('terminal') else node)
            builder.append('\n')

            if tree.getChildCount() > 0:
                children = []
                for i in range(0, tree.getChildCount()):
                    children.append(tree.getChild(i))
                child_list_stack.append(children)



@app.command()
def main(
    input_file: Path = typer.Option(None, "-i", "--input", help="Input .atomic file"),
    output_file: Path = typer.Option("examples/atomic/output.ast", "-o", "--output", help="Output file with parse tree"),
):
    
    with open(input_file, encoding="utf-8") as f:
        source = f.read()
        
    stream = antlr4.InputStream(source)
    lexer = AtomicLexer(stream)
    parser = AtomicParser(antlr4.CommonTokenStream(lexer))
    # tree = parser.program()
    tree = to_string_tree(parser.program(), lexer.symbolicNames)
    
    # builder = _AtomicASTBuilder()
    # ast = builder.visit(tree)
    # ast = analog_operator_canonicalization(ast)
    # tree = tree.model_dump_json(indent=2)

    with open(output_file, 'w') as f:
        f.write(tree)

if __name__ == '__main__':
    app()
    
