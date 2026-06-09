import os
import sys
import mimetypes
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

ROOT = r'd:\Desktop\ai-ml-capstone'
PORT = 8080

class ThreadedRangeHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Prevent noisy logs from cluttering, but print errors
        if "404" in fmt or "500" in fmt:
            print(f'  {self.address_string()} {fmt % args}', file=sys.stderr)

    def do_GET(self):
        try:
            path = self.path.split('?')[0]
            rel = path.lstrip('/').replace('/', os.sep)
            filepath = os.path.join(ROOT, rel)
            
            if os.path.isdir(filepath):
                filepath = os.path.join(filepath, 'index.html')
            
            if not os.path.isfile(filepath):
                self.send_error(404)
                return
            
            mime, _ = mimetypes.guess_type(filepath)
            mime = mime or 'application/octet-stream'
            file_size = os.path.getsize(filepath)
            
            # Simple CORS headers
            self.send_response_only(200 if not self.headers.get('Range') else 206)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Accept-Ranges', 'bytes')
            
            range_header = self.headers.get('Range')
            if range_header and range_header.startswith('bytes='):
                byte_range = range_header.split('=')[1]
                start_str, end_str = byte_range.split('-')
                
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1
                
                self.send_response(206)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                self.send_header('Content-Length', str(length))
                self.end_headers()
                
                with open(filepath, 'rb') as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            else:
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(file_size))
                self.end_headers()
                
                with open(filepath, 'rb') as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
        except (ConnectionResetError, ConnectionAbortedError):
            # Client closed the connection early (e.g. during a seek/drag event)
            # This is expected behavior for HTML5 audio elements and should be handled silently.
            pass
        except Exception as e:
            try:
                self.send_error(500, str(e))
            except Exception:
                pass

if __name__ == '__main__':
    server = ThreadingHTTPServer(('localhost', PORT), ThreadedRangeHandler)
    print(f'Serving d:/Desktop/ai-ml-capstone on ThreadingHTTPServer')
    print(f'Open: http://localhost:{PORT}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
