"""File and configuration management utilities"""

import json
from pathlib import Path

# ===== Configuration Management =====

def save_last_directory(directory):
    """Save last opened directory to config file
    
    Args:
        directory: Path object or string
    """
    import json
    from pathlib import Path
    
    config_file = Path.home() / '.yaaat_config.json'
    try:
        # Load existing config if it exists
        config = {}
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        # Update last directory
        config['last_directory'] = str(directory)
        
        # Save config
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save config: {e}")

def load_last_directory():
    """Load last opened directory from config file
    
    Returns:
        Path object if valid directory exists, None otherwise
    """
    import json
    from pathlib import Path
    
    config_file = Path.home() / '.yaaat_config.json'
    try:
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                last_dir = config.get('last_directory', '')
                if last_dir:
                    last_dir = Path(last_dir)
                    if last_dir.exists() and last_dir.is_dir():
                        return last_dir
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
    return None