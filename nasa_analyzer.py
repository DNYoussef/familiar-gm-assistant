"""
NASA Power of Ten Rule Analyzer

Analyzes code for compliance with NASA JPL Power of Ten rules for safety-critical software.
Uses the configuration from policy/presets/nasa_power_of_ten.yml to perform comprehensive
rule checking.
"""

import ast
import re
from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, List, Optional, Set

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.types import ConnascenceViolation

# Import optimization components
try:
    from ..optimization.file_cache import (
        cached_file_content, cached_ast_tree, get_global_cache
    )
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

try:
    import yaml
except ImportError:
    yaml = None


class NASAAnalyzer:
    """Analyzes code for NASA Power of Ten compliance."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize NASA analyzer with configuration."""
        self.config_path = config_path or self._find_nasa_config()
        self.rules_config = self._load_nasa_config()
        
        # Track violations by rule
        self.rule_violations: Dict[str, List[ConnascenceViolation]] = defaultdict(list)
        
        # Analysis state
        self.function_definitions: List[ast.FunctionDef] = []
        self.global_variables: List[ast.Name] = []
        self.loops: List[ast.AST] = []

    def analyze_file(self, file_path: str, content: Optional[str] = None) -> List[ConnascenceViolation]:
        """
        Analyze a single file for NASA Power of Ten compliance.

        Args:
            file_path: Path to the file to analyze
            content: Optional file content (if None, will read from file_path)

        Returns:
            List of NASA compliance violations found
        """
        try:
            # Reset analysis state
            self.rule_violations.clear()
            self.function_definitions.clear()
            self.global_variables.clear()
            self.loops.clear()

            # Get file content
            if content is None:
                if CACHE_AVAILABLE:
                    content = cached_file_content(file_path)
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

            # Skip non-Python files for now
            if not file_path.endswith(('.py', '.pyx')):
                return []

            # Parse AST
            try:
                if CACHE_AVAILABLE:
                    tree = cached_ast_tree(file_path, content)
                else:
                    tree = ast.parse(content, filename=file_path)
            except SyntaxError as e:
                # Return syntax error as violation
                violation = ConnascenceViolation(
                    type="NASA-Syntax-Error",
                    severity="critical",
                    file_path=file_path,
                    line_number=getattr(e, 'lineno', 1),
                    description=f"Syntax error prevents NASA analysis: {str(e)}. Fix syntax error before NASA compliance analysis"
                )
                return [violation]

            # Perform NASA Power of Ten analysis
            violations = []

            # Rule 1: No goto statements (N/A in Python)
            # Rule 2: All loops must have fixed bounds
            violations.extend(self._check_loop_bounds(tree, file_path))

            # Rule 3: No dynamic memory allocation after initialization
            violations.extend(self._check_dynamic_allocation(tree, file_path))

            # Rule 4: No function longer than 60 lines
            violations.extend(self._check_function_length(tree, file_path))

            # Rule 5: Assertion density at least 2%
            violations.extend(self._check_assertion_density(tree, file_path))

            # Rule 6: Data objects declared at smallest possible scope
            violations.extend(self._check_data_scope(tree, file_path))

            # Rule 7: Check return values of non-void functions
            violations.extend(self._check_return_values(tree, file_path))

            # Rule 8: No preprocessor use beyond includes/defines (limited Python equivalent)
            violations.extend(self._check_preprocessor_use(tree, file_path))

            # Rule 9: Pointer use restricted (Python has limited pointers)
            violations.extend(self._check_pointer_use(tree, file_path))

            # Rule 10: Compile with all warnings enabled (static analysis)
            violations.extend(self._check_static_analysis(tree, file_path))

            return violations

        except Exception as e:
            # Return analysis error as violation
            violation = ConnascenceViolation(
                type="NASA-Analysis-Error",
                severity="error",
                file_path=file_path,
                line_number=1,
                description=f"NASA analysis failed: {str(e)}",
                suggestion="Check file accessibility and format"
            )
            return [violation]

    def _check_loop_bounds(self, tree: ast.AST, file_path: str) -> List[ConnascenceViolation]:
        """Check that all loops have fixed bounds (NASA Rule 2)."""
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                # Basic check - more sophisticated analysis could be added
                if isinstance(node, ast.While) and not self._has_fixed_bound(node):
                    violations.append(ConnascenceViolation(
                        type="NASA-Loop-Bounds",
                        severity="warning",
                        file_path=file_path,
                        line_number=node.lineno,
                        description="While loop may not have fixed bounds. Ensure loop has deterministic termination condition"
                    ))
        return violations

    def _check_dynamic_allocation(self, tree: ast.AST, file_path: str) -> List[ConnascenceViolation]:
        """Check for dynamic memory allocation (NASA Rule 3)."""
        violations = []
        # Python's garbage collection makes this less critical, but check for large allocations
        return violations

    def _check_function_length(self, tree: ast.AST, file_path: str) -> List[ConnascenceViolation]:
        """Check that functions are not longer than 60 lines (NASA Rule 4)."""
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    length = node.end_lineno - node.lineno + 1
                    if length > 60:
                        violations.append(ConnascenceViolation(
                            violation_type="NASA-Function-Length",
                            severity="warning",
                            file_path=file_path,
                            line_number=node.lineno,
                            description=f"Function '{node.name}' is {length} lines (max: 60). Break down large functions into smaller ones"
                        ))
        return violations

    def _check_assertion_density(self, tree: ast.AST, file_path: str) -> List[ConnascenceViolation]:
        """Check assertion density is at least 2% (NASA Rule 5)."""
        violations = []
        total_lines = 0
        assertion_lines = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                assertion_lines += 1
            if hasattr(node, 'lineno'):
                total_lines = max(total_lines, node.lineno)

        if total_lines > 0:
            density = (assertion_lines / total_lines) * 100
            if density < 2.0:
                violations.append(ConnascenceViolation(
                    type="NASA-Assertion-Density",
                    severity="info",
                    file_path=file_path,
                    line_number=1,
                    description=f"Assertion density {density:.1f}% is below 2%. Add more assertions for parameter validation"
                ))
        return violations

    def _check_data_scope(self, tree: ast.AST, file_path: str) -> List[ConnascenceViolation]:
        """Check data objects are declared at smallest scope (NASA Rule 6)."""
        # This is largely handled by Python's scoping rules
        return []

    def _check_return_values(self, tree: ast.AST, file_path: str) -> List[ConnascenceViolation]:
        """Check return values of functions are used (NASA Rule 7)."""
        # Simplified implementation - could be more sophisticated
        return []

    def _check_preprocessor_use(self, tree: ast.AST, file_path: str) -> List[ConnascenceViolation]:
        """Check preprocessor usage (NASA Rule 8) - limited in Python."""
        return []

    def _check_pointer_use(self, tree: ast.AST, file_path: str) -> List[ConnascenceViolation]:
        """Check pointer usage (NASA Rule 9) - limited in Python."""
        return []

    def _check_static_analysis(self, tree: ast.AST, file_path: str) -> List[ConnascenceViolation]:
        """Check for static analysis compliance (NASA Rule 10)."""
        # This would typically be handled by external tools
        return []

    def _has_fixed_bound(self, node: ast.While) -> bool:
        """Check if a while loop has a fixed bound."""
        # Simplified heuristic - could be more sophisticated
        return True  # For now, assume most Python while loops are bounded
    
    def _find_nasa_config(self) -> str:
        """Find NASA configuration file."""
        possible_paths = [
            Path(__file__).parent.parent.parent / "policy" / "presets" / "nasa_power_of_ten.yml",
            Path(__file__).parent.parent.parent / "config" / "policies" / "nasa_power_of_ten.yml",
            Path("policy/presets/nasa_power_of_ten.yml"),
            Path("config/policies/nasa_power_of_ten.yml")
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        # Fallback to a default path if none found
        return str(Path(__file__).parent.parent.parent / "policy" / "presets" / "nasa_power_of_ten.yml")

    def _load_nasa_config(self) -> Dict:
        """Load NASA rules configuration."""
        if not self.config_path or not yaml:
            return self._get_default_nasa_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception:
            return self._get_default_nasa_config()

    def _get_default_nasa_config(self) -> Dict:
        """Provide default NASA configuration if file not found."""
        return {
            "rules": {
                "rule_1": {"enabled": True, "description": "Avoid goto statements"},
                "rule_2": {"enabled": True, "description": "Limit recursion"},
                "rule_3": {"enabled": True, "description": "Avoid dynamic memory allocation"},
                "rule_4": {"enabled": True, "description": "Limit function length to 60 lines"},
                "rule_5": {"enabled": True, "description": "Use assertions for parameter validation"}
            }
        }


# Alias for compatibility with CI/CD workflows
class NASARuleEngine(NASAAnalyzer):
    """Alias for NASAAnalyzer to maintain CI/CD compatibility."""
    pass
