"""
Red Team Testing Suite for SynthID Code Watermarking
Tests various adversarial attacks to find watermark breaking points
"""

import re
import ast
import astor
import random
from typing import List, Tuple, Dict


class AdversarialTransformer:
    """Applies various code transformations to test watermark robustness"""

    def __init__(self, code: str):
        self.original_code = code
        self.code = code

    def extract_variables(self) -> List[str]:
        """Extract all variable names from code"""
        try:
            tree = ast.parse(self.code)
            variables = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    # Skip built-in names
                    if node.id not in ['True', 'False', 'None', 'print', 'input', 'len', 'range', 'int', 'str', 'list', 'dict']:
                        variables.add(node.id)

            return list(variables)
        except SyntaxError:
            # Fallback to regex if AST parsing fails
            return re.findall(r'\b([a-z_][a-z0-9_]*)\b', self.code)

    def rename_variables(self, num_renames: int = None, strategy: str = 'random') -> str:
        """
        Rename variables in the code

        Args:
            num_renames: Number of variables to rename (None = all)
            strategy: 'random' (random names), 'semantic' (meaningful), 'obfuscate' (single chars)
        """
        try:
            tree = ast.parse(self.code)
            variables = self.extract_variables()

            if not variables:
                return self.code

            # Select variables to rename
            if num_renames is None or num_renames >= len(variables):
                vars_to_rename = variables
            else:
                vars_to_rename = random.sample(variables, min(num_renames, len(variables)))

            # Generate new names based on strategy
            rename_map = {}
            for i, var in enumerate(vars_to_rename):
                if strategy == 'random':
                    new_name = f"var_{random.randint(1000, 9999)}"
                elif strategy == 'obfuscate':
                    # Single character names
                    new_name = chr(97 + (i % 26)) + (str(i // 26) if i >= 26 else '')
                elif strategy == 'semantic':
                    # More descriptive names
                    new_name = f"variable_{i}"
                else:
                    new_name = f"renamed_{var}"

                rename_map[var] = new_name

            # Apply renaming using AST transformer
            class VariableRenamer(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id in rename_map:
                        node.id = rename_map[node.id]
                    return node

            renamer = VariableRenamer()
            new_tree = renamer.visit(tree)

            return astor.to_source(new_tree)

        except Exception as e:
            # Fallback to simple regex replacement
            modified = self.code
            variables = self.extract_variables()

            if num_renames:
                variables = random.sample(variables, min(num_renames, len(variables)))

            for i, var in enumerate(variables):
                if strategy == 'obfuscate':
                    new_name = chr(97 + (i % 26))
                else:
                    new_name = f"var_{i}"
                # Use word boundary to avoid partial replacements
                modified = re.sub(r'\b' + re.escape(var) + r'\b', new_name, modified)

            return modified

    def add_whitespace(self, intensity: int = 1) -> str:
        """Add random whitespace/newlines"""
        lines = self.code.split('\n')
        modified_lines = []

        for line in lines:
            modified_lines.append(line)
            if intensity > 1 and random.random() < 0.3:
                modified_lines.append('')  # Add blank line

        return '\n'.join(modified_lines)

    def remove_whitespace(self) -> str:
        """Remove unnecessary whitespace"""
        # Remove blank lines
        lines = [line for line in self.code.split('\n') if line.strip()]
        return '\n'.join(lines)

    def add_comments(self, num_comments: int = 5) -> str:
        """Add random comments to code"""
        lines = self.code.split('\n')
        comment_templates = [
            "# Processing data",
            "# Calculate result",
            "# Helper function",
            "# TODO: optimize",
            "# Variable initialization"
        ]

        modified_lines = []
        for line in lines:
            modified_lines.append(line)
            if num_comments > 0 and random.random() < 0.2:
                modified_lines.append('    ' + random.choice(comment_templates))
                num_comments -= 1

        return '\n'.join(modified_lines)

    def reformat_code(self, style: str = 'compact') -> str:
        """Reformat code style (compact vs expanded)"""
        try:
            tree = ast.parse(self.code)

            if style == 'compact':
                # Minimize whitespace
                return astor.to_source(tree).replace('\n\n', '\n')
            else:
                # Expand whitespace
                formatted = astor.to_source(tree)
                # Add extra spacing
                formatted = formatted.replace('\n', '\n\n')
                return formatted

        except:
            return self.code

    def rename_functions(self, num_renames: int = None) -> str:
        """Rename function definitions"""
        try:
            tree = ast.parse(self.code)

            # Find all function names
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name != 'solve':  # Don't rename main solve function
                        functions.append(node.name)

            if not functions:
                return self.code

            if num_renames is None:
                funcs_to_rename = functions
            else:
                funcs_to_rename = random.sample(functions, min(num_renames, len(functions)))

            # Create rename mapping
            rename_map = {func: f"func_{i}" for i, func in enumerate(funcs_to_rename)}

            class FunctionRenamer(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    if node.name in rename_map:
                        node.name = rename_map[node.name]
                    self.generic_visit(node)
                    return node

                def visit_Call(self, node):
                    if isinstance(node.func, ast.Name) and node.func.id in rename_map:
                        node.func.id = rename_map[node.func.id]
                    self.generic_visit(node)
                    return node

            renamer = FunctionRenamer()
            new_tree = renamer.visit(tree)

            return astor.to_source(new_tree)

        except:
            return self.code

    def apply_combined_attack(self, var_renames: int = 0, func_renames: int = 0,
                             add_comments_count: int = 0, reformat: bool = False) -> str:
        """Apply multiple transformations"""
        code = self.code

        if var_renames > 0:
            transformer = AdversarialTransformer(code)
            code = transformer.rename_variables(num_renames=var_renames)

        if func_renames > 0:
            transformer = AdversarialTransformer(code)
            code = transformer.rename_functions(num_renames=func_renames)

        if add_comments_count > 0:
            transformer = AdversarialTransformer(code)
            code = transformer.add_comments(num_comments=add_comments_count)

        if reformat:
            transformer = AdversarialTransformer(code)
            code = transformer.reformat_code()

        return code


def generate_attack_suite(code: str) -> List[Tuple[str, str, Dict]]:
    """
    Generate a suite of adversarial attacks with increasing intensity

    Returns: List of (attack_name, modified_code, metadata)
    """
    attacks = []

    # Baseline - no modification
    attacks.append(("baseline", code, {"intensity": 0}))

    transformer = AdversarialTransformer(code)
    variables = transformer.extract_variables()
    num_vars = len(variables)

    # Variable renaming attacks - increasing intensity
    for intensity in [1, 5, 10, 25, 50, 100]:
        if intensity <= num_vars:
            modified = AdversarialTransformer(code).rename_variables(num_renames=intensity, strategy='obfuscate')
            attacks.append((f"rename_{intensity}_vars", modified, {"intensity": intensity, "type": "variable_rename"}))

    # Rename ALL variables
    if num_vars > 0:
        modified = AdversarialTransformer(code).rename_variables(num_renames=None, strategy='obfuscate')
        attacks.append((f"rename_all_{num_vars}_vars", modified, {"intensity": num_vars, "type": "variable_rename"}))

    # Function renaming
    modified = AdversarialTransformer(code).rename_functions()
    attacks.append(("rename_functions", modified, {"intensity": 5, "type": "function_rename"}))

    # Whitespace attacks
    modified = AdversarialTransformer(code).add_whitespace(intensity=2)
    attacks.append(("add_whitespace", modified, {"intensity": 3, "type": "whitespace"}))

    modified = AdversarialTransformer(code).remove_whitespace()
    attacks.append(("remove_whitespace", modified, {"intensity": 3, "type": "whitespace"}))

    # Comment injection
    modified = AdversarialTransformer(code).add_comments(num_comments=10)
    attacks.append(("add_comments", modified, {"intensity": 2, "type": "comments"}))

    # Reformatting
    modified = AdversarialTransformer(code).reformat_code(style='compact')
    attacks.append(("reformat_compact", modified, {"intensity": 2, "type": "formatting"}))

    # Combined attacks - increasing severity
    modified = AdversarialTransformer(code).apply_combined_attack(
        var_renames=min(3, num_vars), add_comments_count=5
    )
    attacks.append(("combined_light", modified, {"intensity": 4, "type": "combined"}))

    modified = AdversarialTransformer(code).apply_combined_attack(
        var_renames=min(10, num_vars), func_renames=2, add_comments_count=10
    )
    attacks.append(("combined_medium", modified, {"intensity": 6, "type": "combined"}))

    modified = AdversarialTransformer(code).apply_combined_attack(
        var_renames=num_vars, func_renames=5, add_comments_count=20, reformat=True
    )
    attacks.append(("combined_heavy", modified, {"intensity": 9, "type": "combined"}))

    return attacks
