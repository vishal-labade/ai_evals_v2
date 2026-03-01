import argparse
import requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:11434")
    args = ap.parse_args()

    url = args.host.rstrip("/") + "/api/tags"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    models = [m.get("name") for m in data.get("models", [])]
    print("Ollama OK. Models:")
    for m in models:
        print(" -", m)

if __name__ == "__main__":
    main()