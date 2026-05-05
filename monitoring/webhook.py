"""
Alert Webhook Server
Receives alerts from Alertmanager and triggers GitHub Actions workflow.
"""

import os
import json
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler


REPO = "zunaira-ahmd/FraudOps"


class WebhookHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        if self.path != "/alert":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
            alerts = data.get("alerts", [])

            for alert in alerts:
                name = alert.get("labels", {}).get("alertname", "unknown")
                status = alert.get("status", "firing")

                if status != "firing":
                    continue

                print(f"Alert received: {name}")

                if name in ("FraudRecallBelowThreshold", "FeatureDriftHigh"):
                    trigger_reason = (
                        "recall_dropped" if name == "FraudRecallBelowThreshold"
                        else "drift_detected"
                    )
                    print(f"Triggering retraining: {trigger_reason}")
                    subprocess.run([
                        "gh", "workflow", "run", "ci-cd.yml",
                        "-R", REPO,
                        "-f", f"trigger_reason={trigger_reason}",
                    ])

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        except Exception as e:
            print(f"Webhook error: {e}")
            self.send_response(500)
            self.end_headers()

    def log_message(self, format, *args):
        print(f"[webhook] {args[0]} {args[1]} {args[2]}")


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 5001), WebhookHandler)
    print("Webhook server running on port 5001")
    server.serve_forever()
