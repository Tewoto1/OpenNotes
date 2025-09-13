import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import difflib
import os

tracked_file_types = ['.txt', '.md', '.py', '.java', '.js', '.html', '.css']
workspace_dir = os.path.basename(os.getcwd())

class Index:
    def __init__(self):
        self.structure = {}

    def build_user_index(self):
        self.structure[workspace_dir] = {}
        for root, dirs, files in os.walk('.'):
            for file in files:
                if any(file.endswith(ext) for ext in tracked_file_types):
                    file_path = os.path.relpath(os.path.join(root, file), '.')
                    with open(file_path, 'r') as f:
                        lines = f.read().splitlines()
                        lines = list(filter(lambda line: line.strip(), lines))
                        self.structure[workspace_dir][file_path] = [{'text':line} for line in lines] 

    def update_file_index(self, file_path):
        lines = open(file_path, 'r').read().splitlines()
        lines = list(filter(lambda line: line.strip(), lines)) # ignore empty lines
        rel_file_path = os.path.relpath(file_path, '.')
        file_structure = self.structure[workspace_dir][rel_file_path]
        o_lines = [atom['text'] for atom in file_structure]
        matcher = difflib.SequenceMatcher(None, o_lines, lines)
        new_file_structure = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                for i in range(i1, i2):
                    new_file_structure.append({'text': lines[i - i1 + j1]})
            elif tag == 'delete':
                pass
            elif tag == 'insert':
                for j in range(j1, j2):
                    new_file_structure.append({'text': lines[j]})
            elif tag == 'equal':
                new_file_structure.extend(file_structure[i1:i2])
        self.structure[workspace_dir][rel_file_path] = new_file_structure

index = Index()
index.build_user_index()

class Handler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory and any(event.src_path.endswith(ext) for ext in tracked_file_types):
            index.update_file_index(event.src_path)


def start_watcher(path="."):
    event_handler = Handler()
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=True)
    observer.start()
    return observer


def query(msg):
    pass

def interactive_loop():
    while True:
        try:
            cmd = input("> ").split()
            if not cmd:
                continue
            if cmd[0].strip() == 'quit' or cmd[0].strip() == 'q':
                break
            elif cmd[0].strip() == 'help' or cmd[0].strip() == 'h':
                print("Available commands: index, query, help, quit")
            elif cmd[0].strip() == 'index' or cmd[0].strip() == 'i':
                index.build_user_index()
            elif cmd[0].strip() == 'query' or cmd[0].strip() == 'q':
                if len(cmd) < 2:
                    print("Usage: query <message>")
                else:
                    msg = " ".join(cmd[1:])
                    query(msg)
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    observer = start_watcher()  # background watcher thread
    try:
        interactive_loop()  # foreground REPL
    finally:
        observer.stop()
        observer.join()

