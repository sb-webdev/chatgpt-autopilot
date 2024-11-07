#!/usr/bin/env python3
"""
OpenAI API Migration Script

This script helps migrate code from the old OpenAI API (pre-1.0.0) to the new API style.
It handles common patterns but manual review is still recommended.

Usage:
    python openai_upgrader.py <path_to_file>
    python openai_upgrader.py <path_to_directory>
"""

import sys
import re
import os
from pathlib import Path
import ast
from typing import List, Tuple

class OpenAIUpgrader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.content = ""
        
    def read_file(self) -> str:
        """Read the content of the file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.content = f.read()
        return self.content

    def write_file(self, new_content: str):
        """Write the updated content back to the file."""
        backup_path = self.file_path + '.bak'
        # Create backup
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(self.content)
        # Write new content
        with open(self.file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✓ Updated {self.file_path} (backup created at {backup_path})")

    def upgrade_imports(self, content: str) -> str:
        """Update import statements to new style."""
        # Replace simple import
        content = re.sub(
            r'import\s+openai\s*($|\n)',
            'from openai import OpenAI\n',
            content
        )
        
        # Replace selective imports
        content = re.sub(
            r'from\s+openai\s+import\s+(?!OpenAI)(.*?)($|\n)',
            'from openai import OpenAI  # Upgraded: selective imports are now accessed via client\n',
            content
        )
        
        return content

    def upgrade_api_key_setting(self, content: str) -> str:
        """Update API key configuration to client-based approach."""
        # Find API key assignments
        api_key_pattern = r'openai\.api_key\s*=\s*(.*?)($|\n)'
        matches = re.finditer(api_key_pattern, content)
        
        for match in matches:
            api_key_value = match.group(1)
            # Add client initialization after the API key setting
            replacement = f'client = OpenAI(api_key={api_key_value})'
            content = content.replace(match.group(0), replacement)
        
        return content

    def upgrade_completion_calls(self, content: str) -> str:
        """Update completion API calls to new style."""
        # Update standard completions
        content = re.sub(
            r'openai\.Completion\.create\(',
            'client.completions.create(',
            content
        )
        
        # Update chat completions
        content = re.sub(
            r'openai\.ChatCompletion\.create\(',
            'client.chat.completions.create(',
            content
        )
        
        return content

    def upgrade_other_api_calls(self, content: str) -> str:
        """Update other API calls to use client."""
        patterns = {
            r'openai\.Image\.create': 'client.images.generate',
            r'openai\.Engine\.list': 'client.models.list',
            r'openai\.File\.create': 'client.files.create',
            r'openai\.FineTune\.create': 'client.fine_tunes.create',
            r'openai\.Moderation\.create': 'client.moderations.create',
        }
        
        for old, new in patterns.items():
            content = re.sub(old, new, content)
        
        return content

    def add_client_initialization(self, content: str) -> str:
        """Add client initialization if not present."""
        if 'client = OpenAI(' not in content and 'OpenAI(' not in content:
            # Add after imports but before other code
            import_section_end = 0
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip() and not (
                    line.startswith('import') or 
                    line.startswith('from') or 
                    line.startswith('#')
                ):
                    import_section_end = i
                    break
            
            lines.insert(import_section_end, '\nclient = OpenAI()')
            content = '\n'.join(lines)
        
        return content

    def process_file(self) -> bool:
        """Process the file and apply all upgrades."""
        try:
            content = self.read_file()
            
            # Skip if file doesn't use OpenAI
            if 'openai' not in content:
                print(f"Skipping {self.file_path} (no OpenAI usage found)")
                return False
            
            # Apply transformations
            content = self.upgrade_imports(content)
            content = self.upgrade_api_key_setting(content)
            content = self.upgrade_completion_calls(content)
            content = self.upgrade_other_api_calls(content)
            content = self.add_client_initialization(content)
            
            # Write updated content
            self.write_file(content)
            return True
            
        except Exception as e:
            print(f"Error processing {self.file_path}: {str(e)}")
            return False

def process_directory(directory_path: str) -> Tuple[int, int]:
    """Process all Python files in a directory."""
    processed = 0
    updated = 0
    
    for path in Path(directory_path).rglob('*.py'):
        if '.venv' not in str(path) and 'env' not in str(path):
            processed += 1
            upgrader = OpenAIUpgrader(str(path))
            if upgrader.process_file():
                updated += 1
                
    return processed, updated

def main():
    if len(sys.argv) != 2:
        print("Usage: python openai_upgrader.py <path_to_file_or_directory>")
        sys.exit(1)

    path = sys.argv[1]
    
    if os.path.isfile(path):
        upgrader = OpenAIUpgrader(path)
        upgrader.process_file()
    elif os.path.isdir(path):
        processed, updated = process_directory(path)
        print(f"\nProcessed {processed} files, updated {updated} files")
    else:
        print(f"Error: {path} not found")
        sys.exit(1)

if __name__ == "__main__":
    main()