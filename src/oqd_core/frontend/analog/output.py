from .AnalogLexer import AnalogLexer
from .AnalogParser import AnalogParser
import sys
from pathlib import Path
import antlr4
import typer

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
    input_file: Path = typer.Option(None, "-i", "--input", help="Input .analog file"),
    output_file: Path = typer.Option(None, "-o", "--output", help="Output file with parse tree"),
):

    input_stream = antlr4.FileStream(str(input_file), encoding="utf-8")
    lexer = AnalogLexer(input_stream)
    parser = AnalogParser(antlr4.CommonTokenStream(lexer))
    tree = to_string_tree(parser.program(), lexer.symbolicNames)

    with open(output_file, 'w') as f:
        f.write(tree)

if __name__ == '__main__':
    app()