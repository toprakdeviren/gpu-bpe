// ─── Logger Class ───
export class Logger {
    constructor(logElement) {
        this.logElement = logElement;
        this.originalConsoleLog = console.log;
        this.interceptConsole();
    }

    log(message) {
        this.logElement.textContent += message + '\n';
        this.logElement.scrollTop = this.logElement.scrollHeight;
    }

    interceptConsole() {
        console.log = (...args) => {
            this.originalConsoleLog(...args);
            this.log(args.join(' '));
        };
    }
}
