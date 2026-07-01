module.exports = {
  apps: [{
    name: 'claude-proxy',
    script: './src/index.ts',
    interpreter: 'npx',
    interpreter_args: 'tsx',
    cwd: __dirname,
    env: {
      NODE_ENV: 'production',
    },
    instances: 1,
    exec_mode: 'fork',
    watch: false,
    max_memory_restart: '500M',
    error_file: './logs/err.log',
    out_file: './logs/out.log',
    log_file: './logs/combined.log',
    time: true,
    kill_timeout: 5000,
    shutdown_with_message: true,
  }],
};
