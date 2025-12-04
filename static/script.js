// ATM Security System

class ATMSystem {
    constructor() {
        this.currentScreen = 'welcome';
        this.pinValue = '';
        this.balance = 5000.00;
        this.withdrawalAmount = 0;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.startStatusPolling();
    }

    setupEventListeners() {
        // Insert Card
        document.getElementById('insert-card-btn')?.addEventListener('click', () => this.insertCard());

        // Keypad
        document.querySelectorAll('.key').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const key = e.target.dataset.key;
                this.handleKey(key);
            });
        });

        // Menu buttons
        document.getElementById('withdrawal-btn')?.addEventListener('click', () => this.selectWithdrawal());
        document.getElementById('balance-btn')?.addEventListener('click', () => this.showBalance());
        document.getElementById('exit-btn')?.addEventListener('click', () => this.reset());

        // Quick amounts
        document.querySelectorAll('.quick-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.getElementById('amount-input').value = e.target.dataset.amount;
            });
        });

        // Process button
        document.getElementById('process-btn')?.addEventListener('click', () => this.processAction());

        // Cancel button
        document.getElementById('cancel-btn')?.addEventListener('click', () => this.reset());

        // Receipt choice buttons
        document.getElementById('receipt-yes-btn')?.addEventListener('click', () => this.printReceiptChoice(true));
        document.getElementById('receipt-no-btn')?.addEventListener('click', () => this.printReceiptChoice(false));
    }

    showScreen(screenName) {
        document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
        document.getElementById(`${screenName}-screen`)?.classList.add('active');
        this.currentScreen = screenName;
        this.updateButtons();
    }

    updateButtons() {
        const insertBtn = document.getElementById('insert-card-btn');
        const processBtn = document.getElementById('process-btn');

        insertBtn.style.display = this.currentScreen === 'welcome' ? 'block' : 'none';
        processBtn.style.display = ['pin', 'withdrawal'].includes(this.currentScreen) ? 'block' : 'none';

        if (this.currentScreen === 'pin') processBtn.textContent = 'Enter PIN';
        if (this.currentScreen === 'withdrawal') processBtn.textContent = 'Withdraw';
    }

    async insertCard() {
        // Hide any previous receipt
        const receiptPaper = document.getElementById('receipt-paper');
        receiptPaper.classList.remove('printing');
        receiptPaper.style.display = 'none';
        receiptPaper.style.bottom = '-100%';
        
        const cardLed = document.getElementById('card-led');
        const cardAnim = document.getElementById('card-animation');

        cardLed.classList.add('active');
        cardAnim.classList.add('inserting');

        await new Promise(r => setTimeout(r, 2000));

        try {
            const res = await fetch('/api/insert_card', { method: 'POST' });
            const data = await res.json();

            if (data.success) {
                this.showScreen('pin');
                this.pinValue = '';
                document.getElementById('pin-input').value = '';
            } else {
                this.showError(data.message);
                cardLed.classList.remove('active');
            }
        } catch (err) {
            this.showError('System error');
            cardLed.classList.remove('active');
        }

        cardAnim.classList.remove('inserting');
    }

    handleKey(key) {
        if (this.currentScreen !== 'pin') return;

        const input = document.getElementById('pin-input');

        if (key === 'clear') {
            this.pinValue = '';
            input.value = '';
        } else if (key === 'cancel') {
            this.reset();
        } else if (key === 'enter') {
            this.enterPin();
        } else if (this.pinValue.length < 4) {
            this.pinValue += key;
            input.value = '•'.repeat(this.pinValue.length);
        }
    }

    async enterPin() {
        if (this.pinValue.length !== 4) {
            this.showError('Enter 4-digit PIN');
            return;
        }

        try {
            const res = await fetch('/api/enter_pin', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pin: this.pinValue })
            });
            const data = await res.json();

            if (data.success) {
                this.showScreen('menu');
            } else {
                this.showError(data.message);
                this.pinValue = '';
                document.getElementById('pin-input').value = '';
            }
        } catch (err) {
            this.showError('System error');
        }
    }

    async selectWithdrawal() {
        try {
            await fetch('/api/select_withdrawal', { method: 'POST' });
            this.showScreen('withdrawal');
            document.getElementById('amount-input').value = '';
        } catch (err) {
            this.showError('System error');
        }
    }

    showBalance() {
        alert(`Balance: $${this.balance.toFixed(2)}`);
    }

    async processAction() {
        if (this.currentScreen === 'pin') {
            await this.enterPin();
        } else if (this.currentScreen === 'withdrawal') {
            await this.processWithdrawal();
        }
    }

    async processWithdrawal() {
        const amount = parseFloat(document.getElementById('amount-input').value);

        if (!amount || amount <= 0) {
            this.showError('Enter valid amount');
            return;
        }

        if (amount > 1000) {
            this.showError('Max withdrawal: $1,000');
            return;
        }

        this.showScreen('processing');

        try {
            const res = await fetch('/api/withdraw', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ amount })
            });
            const data = await res.json();

            setTimeout(() => {
                if (data.success) {
                    this.balance = data.new_balance;
                    this.withdrawalAmount = amount;
                    this.updateBalance();
                    document.getElementById('balance-complete').textContent = this.balance.toFixed(2);
                    this.showScreen('complete');
                    
                    // Show receipt prompt after 2 seconds
                    setTimeout(() => this.showScreen('receipt-prompt'), 2000);
                } else {
                    this.showError(data.message);
                    this.showScreen('withdrawal');
                }
            }, 2000);
        } catch (err) {
            this.showError('System error');
            this.showScreen('withdrawal');
        }
    }

    printReceiptChoice(wantReceipt) {
        if (wantReceipt) {
            this.printReceipt(this.withdrawalAmount, this.balance);
        }
        
        // Return to welcome screen after 3 seconds
        setTimeout(() => this.reset(), 3000);
    }

    printReceipt(amount, balance) {
        const led = document.getElementById('receipt-led');
        const paper = document.getElementById('receipt-paper');

        // Update receipt content
        document.getElementById('receipt-amount').textContent = amount.toFixed(2);
        document.getElementById('receipt-balance').textContent = balance.toFixed(2);

        // Reset position and show receipt
        paper.style.bottom = '-100%';
        paper.style.display = 'block';
        
        // Activate LED and start printing animation
        led.classList.add('active');
        
        // Small delay to ensure display is set before animation
        setTimeout(() => {
            paper.classList.add('printing');
        }, 50);

        // Deactivate LED after 2 seconds
        setTimeout(() => led.classList.remove('active'), 2000);
        
        // Keep receipt visible for 5 seconds, then hide it
        setTimeout(() => {
            paper.classList.remove('printing');
            paper.style.display = 'none';
            paper.style.bottom = '-100%';
        }, 7000);
    }

    async reset() {
        try {
            await fetch('/api/reset', { method: 'POST' });
        } catch (err) {
            console.error('Reset error:', err);
        }

        // Reset all LEDs and animations
        document.getElementById('card-led').classList.remove('active');
        document.getElementById('receipt-led').classList.remove('active');
        document.getElementById('card-animation').classList.remove('inserting');
        
        // Hide and reset receipt paper
        const receiptPaper = document.getElementById('receipt-paper');
        receiptPaper.classList.remove('printing');
        receiptPaper.style.display = 'none';
        receiptPaper.style.bottom = '-100%';

        this.pinValue = '';
        this.withdrawalAmount = 0;
        this.showScreen('welcome');
    }

    showError(msg) {
        const div = document.createElement('div');
        div.textContent = msg;
        div.style.cssText = `
            position: fixed; top: 20px; right: 20px;
            background: #ef4444; color: white;
            padding: 15px 25px; border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            z-index: 1000; animation: slideIn 0.3s;
        `;
        document.body.appendChild(div);
        setTimeout(() => div.remove(), 3000);
    }

    updateBalance() {
        document.getElementById('balance').textContent = this.balance.toFixed(2);
        document.getElementById('balance-withdrawal').textContent = this.balance.toFixed(2);
    }

    async startStatusPolling() {
        setInterval(async () => {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();

                this.balance = data.balance;
                this.updateBalance();
                this.updateSecurity(data.security);
            } catch (err) {
                console.error('Status error:', err);
            }
        }, 500);
    }

    updateSecurity(sec) {
        // Face count
        document.getElementById('face-count').textContent = sec.face_count;

        // Access status
        const accessDot = document.getElementById('access-dot');
        const accessText = document.getElementById('access-text');
        if (sec.access_granted) {
            accessDot.className = 'status-dot granted';
            accessText.textContent = 'ACCESS GRANTED';
        } else {
            accessDot.className = 'status-dot denied';
            accessText.textContent = 'ACCESS DENIED';
        }

        // Violation
        const violationDot = document.getElementById('violation-dot');
        const violationText = document.getElementById('violation-text');
        const confidence = document.getElementById('confidence');

        if (sec.violation_type) {
            violationDot.className = 'status-dot warning';
            violationText.textContent = sec.violation_type.replace(/_/g, ' ').toUpperCase();

            if (sec.has_mask) {
                confidence.textContent = `${(sec.mask_confidence * 100).toFixed(0)}%`;
            } else if (sec.has_helmet) {
                confidence.textContent = `${(sec.helmet_confidence * 100).toFixed(0)}%`;
            }
        } else {
            violationDot.className = 'status-dot';
            violationText.textContent = 'NO VIOLATIONS';
            confidence.textContent = '0%';
        }

        // System info
        const issue = document.getElementById('issue');
        if (sec.violation_type) {
            issue.textContent = sec.violation_type.replace(/_/g, ' ');
            issue.style.color = '#f59e0b';
        } else {
            issue.textContent = 'None';
            issue.style.color = '#22c55e';
        }

        // Warnings
        let warnings = 0;
        if (sec.has_mask) warnings++;
        if (sec.has_helmet) warnings++;
        if (sec.face_count > 1) warnings++;
        if (sec.face_count === 0) warnings++;
        document.getElementById('warnings').textContent = warnings;

        // Last update
        const now = new Date();
        document.getElementById('last-update').textContent = now.toLocaleTimeString('en-US', {
            hour: 'numeric',
            minute: '2-digit',
            second: '2-digit',
            hour12: true
        });
    }
}

// Add animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
`;
document.head.appendChild(style);

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    new ATMSystem();
});
