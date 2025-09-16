module.exports = {
  apps : [{
    name: "chatBot",
    script: "/media/ce00166324/ab4e2df3-b2cf-4994-970b-9f896a71c3ee/var/www/html/Learning/python/chatBot/venv/bin/python",
    args: "-m uvicorn test:app --host 127.0.0.1 --port 8080",
    watch: false
  }]
}
