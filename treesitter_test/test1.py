from tree_sitter import Language, Parser

Language.build_library(
  # Store the library in the `build` directory
  './my-languages.so',

  # Include one or more languages
  [
    'tree-sitter-cpp',
    'tree-sitter-python'
  ]
)

CPP_LANGUAGE = Language('./my-languages.so', 'cpp')
PY_LANGUAGE = Language('./my-languages.so', 'python')

parser = Parser()
parser.set_language(PY_LANGUAGE)

tree = parser.parse(bytes("""
def foo():
    if bar:
        baz()
""", "utf8"))

print(1)
