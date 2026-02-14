import { STATUS_CLASSES } from '../utils.js';

// ─── Status Manager Class ───
export class StatusManager {
    constructor(statusDot, statusText) {
        this.statusDot = statusDot;
        this.statusText = statusText;
    }

    setStatus(type, message) {
        this.statusDot.className = STATUS_CLASSES.base;
        const statusClasses = STATUS_CLASSES[type];
        if (statusClasses) this.statusDot.classList.add(...statusClasses);
        this.statusText.textContent = message;
    }
}
