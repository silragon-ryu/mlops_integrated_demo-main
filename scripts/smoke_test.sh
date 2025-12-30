#!/bin/bash

# Target URL: Your API service (api) running on port 8000
# We verify the root endpoint or health check
URL="http://localhost:8000/" 

echo "🚀 Running Smoke Test against $URL..."

# Loop to wait for the container to fully initialize (up to 30s)
for i in {1..10}; do
    # curl flags: -s (silent), -o (output to null), -w (write http code)
    HTTP_CODE=$(curl --write-out %{http_code} --silent --output /dev/null $URL)

    # We accept 200 (OK) or 404 (Not Found) as success proofs that the server is responding.
    # (Since we might not have a defined root route, 404 means the server IS running).
    if [ "$HTTP_CODE" -eq 200 ] || [ "$HTTP_CODE" -eq 404 ] || [ "$HTTP_CODE" -eq 405 ]; then
        echo "✅ Smoke Test Passed: Server responded with HTTP $HTTP_CODE"
        exit 0
    fi
    
    echo "⏳ Waiting for service to start... (Attempt $i/10)"
    sleep 3
done

echo "❌ Smoke Test Failed: Server did not respond after 30 seconds."
exit 1