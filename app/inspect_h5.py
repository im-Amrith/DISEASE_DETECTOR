import h5py
import json
import os

def inspect_h5():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(working_dir, "trained_model", "Disease.h5")
    
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return

    try:
        with h5py.File(model_path, 'r') as f:
            print("Keys in H5 file:", list(f.keys()))
            
            if 'model_config' in f.attrs:
                config_str = f.attrs['model_config']
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                config = json.loads(config_str)
                
                print("\nModel Config Found!")
                print(json.dumps(config, indent=2)[:500] + "...") # Print start of config
                
                if 'config' in config and 'layers' in config['config']:
                    print("\nLayers:")
                    for layer in config['config']['layers']:
                        print(f"- {layer['class_name']}: {layer['config']['name']}")
            else:
                print("No model_config found in attributes.")
                
    except Exception as e:
        print(f"Error inspecting H5: {e}")

if __name__ == "__main__":
    inspect_h5()
