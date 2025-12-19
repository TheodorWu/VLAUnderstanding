from initializer import Initializer

def main():
    config = {
        "method": {"type": "example_method", "params": {}},
        "model": {"type": "example_model", "params": {}}
    }
    initializer = Initializer(config)
    initializer.initialize()

if __name__ == "__main__":
    main()
