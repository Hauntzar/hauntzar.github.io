.PHONY: preview-index

preview-index:
	@nohup python3 -m http.server 8000 --directory . > /dev/null 2>&1 &
	@sleep 1
	@open http://localhost:8000/
	@echo "Preview opened at http://localhost:8000/"
	@echo "To stop server: pkill -f 'http.server 8000'"
