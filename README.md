# README.md

# 🛡️ Aegis - Real-time AI Compliance Guardian

Aegis is a groundbreaking AI-powered solution that provides real-time compliance monitoring for high-stakes business conversations. Built for the IBM AI & Automation Unpacked Hackathon, Aegis demonstrates the transformative potential of IBM's Granite model family in enterprise risk management.

## 🌟 Key Features

- **Real-time Speech Transcription**: Leverages IBM's Granite Speech 8B for accurate, low-latency transcription
- **Proactive Compliance Monitoring**: Analyzes conversations as they happen, not after the fact
- **Intelligent Alert System**: Provides instant, constructive suggestions to prevent compliance violations
- **Hybrid Architecture**: Optimizes cost and performance with cloud/local processing
- **Two-Agent Safety System**: WorkerAgent for analysis + GuardianAgent for quality assurance
- **Enterprise-Ready**: Built with trust, scalability, and responsible AI principles

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Audio Input   │────▶│  Watson Cloud    │────▶│   Transcript    │
│  (Microphone)   │     │ (Granite Speech) │     │                 │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit UI  │◀────│  Guardian Agent  │◀────│  Worker Agent   │
│ (Alerts/Guides) │     │(granite-guardian)│     │(Granite 4.0 Tiny)│
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │   RAG Pipeline  │
                                                  │ (FAISS + Docs)  │
                                                  └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- IBM watsonx.ai account with hackathon access
- Microphone for audio input
- 8GB+ RAM for local model execution

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-team/aegis.git
   cd aegis
   ```

2. **Run setup script**
   ```bash
   chmod +x scripts/setup_environment.sh
   ./scripts/setup_environment.sh
   ```

3. **Configure credentials**
   ```bash
   # Edit .env file with your IBM credentials
   nano .env
   ```

4. **Test connections**
   ```bash
   python scripts/test_connections.py
   ```

5. **Prepare compliance data**
   ```bash
   python scripts/prepare_data.py
   ```

6. **Launch Aegis**
   ```bash
   python app.py
   ```

## 🎯 Usage

1. Open http://localhost:8501 in your browser
2. Click "Start Recording" to begin monitoring
3. Speak naturally - Aegis will transcribe in real-time
4. Watch for compliance alerts and suggestions
5. Use the suggested alternatives to maintain compliance

## 📊 Demo Scenarios

### Scenario 1: Investment Guarantee
**Say**: "This fund has delivered 12% returns for five years, so you're guaranteed similar results."
**Aegis Alert**: ⚠️ Avoid guarantee language
**Suggestion**: "While this fund has shown strong historical performance, past results don't guarantee future returns."

### Scenario 2: Risk Disclosure
**Say**: "This is basically a risk-free investment opportunity."
**Aegis Alert**: ⚠️ All investments carry risk
**Suggestion**: "This investment has historically shown lower volatility, though all investments involve some degree of risk."

## 🔧 Technical Stack

- **Cloud**: IBM watsonx.ai with Granite Speech 8B
- **Local**: Ollama with Granite 4.0 Tiny & granite-guardian-8b
- **Framework**: BeeAI Agent Framework
- **UI**: Streamlit
- **RAG**: FAISS + Sentence Transformers
- **Audio**: PyAudio for real-time capture

## 📁 Project Structure

```
aegis/
├── src/
│   ├── agents/         # BeeAI agent implementations
│   ├── audio/          # Audio capture and processing
│   ├── cloud/          # Watson API integration
│   ├── rag/            # RAG pipeline components
│   └── ui/             # Streamlit interface
├── scripts/            # Setup and utility scripts
├── data/              # Compliance documents
└── config/            # Configuration management
```

## 🏆 Hackathon Alignment

Aegis directly addresses the hackathon's core objectives:

- **Streamlines Business Processes**: Real-time compliance monitoring
- **Enhances Efficiency**: Prevents violations before they occur
- **Drives Industry Transformation**: Paradigm shift from reactive to proactive compliance
- **Showcases IBM Technology**: Synergistic use of Granite 3.3 & 4.0 models

## 🤝 Team

- [Your Name] - Architecture & Backend
- [Team Member 2] - AI/ML & RAG Pipeline
- [Team Member 3] - Frontend & UX
- [Team Member 4] - Compliance Research & Testing

## 📜 License

This project is developed for the IBM AI & Automation Unpacked Hackathon.

---

# DEMO_SCRIPT.md

# Aegis Demo Script - Hackathon Presentation

## Pre-Demo Setup
- [ ] Aegis running on http://localhost:8501
- [ ] Microphone tested and working
- [ ] Browser in full-screen mode
- [ ] Terminal ready to show architecture

## Opening (30 seconds)

"Meet Sarah, a top financial advisor managing millions in client assets. Every day, she walks a tightrope—building trust while avoiding costly compliance violations. A single misspoken word can trigger millions in fines and end careers.

Today, we introduce **Aegis**—a real-time AI guardian that transforms compliance from a reactive burden into a proactive advantage."

## Live Demo (2 minutes)

### Part 1: Show the Interface
"Here's Aegis in action. You can see our clean interface with:
- Live transcript powered by IBM's Granite Speech 8B
- Real-time compliance monitoring
- Instant alert system"

### Part 2: Compliance Violation Demo

**[Click Start Recording]**

"Let me play the role of a financial advisor..."

**Say**: "Good morning! Based on yesterday's market analysis, I'm excited to share an opportunity with you."

*[Wait for transcription to appear]*

**Say**: "This emerging markets fund has delivered 15% returns for three years straight, so you're **guaranteed** to see similar results going forward."

**[ALERT APPEARS]**

"There! Aegis caught it immediately. Before I could finish the non-compliant statement, it:
- Flagged the guarantee language
- Explained the compliance risk
- Provided a compliant alternative"

**Read the suggestion**: "Let me rephrase that—while this fund has shown strong historical performance, past results don't guarantee future returns."

### Part 3: Show Technical Excellence

**[Open terminal briefly]**

"What makes Aegis special is our hybrid architecture:
- Granite Speech in the cloud for perfect transcription
- Granite 4.0 Tiny running locally for instant analysis
- BeeAI framework orchestrating intelligent agents
- All for less than $100 in cloud credits!"

## The Innovation (45 seconds)

"Aegis represents a paradigm shift in three ways:

1. **Proactive vs Reactive**: We prevent violations in real-time, not report them days later

2. **Hybrid Intelligence**: We strategically combine IBM's cloud power with edge efficiency

3. **Human Augmentation**: We don't replace professionals—we make them unstoppable

This is exactly what IBM envisioned: AI that drives industry transformation through trust and innovation."

## Business Impact (30 seconds)

"The impact is massive:
- **For Advisors**: Confidence in every conversation
- **For Firms**: Millions saved in compliance costs
- **For Clients**: Better, safer financial guidance

Aegis turns compliance from a cost center into a competitive advantage."

## Closing (15 seconds)

"Aegis—built with IBM's Granite models, powered by innovation, designed for transformation.

Thank you!"

---

## Q&A Prep

**Q: How does it handle different languages?**
A: "Granite Speech 8B supports multiple languages. Our architecture allows easy language switching."

**Q: What about data privacy?**
A: "Our hybrid design keeps sensitive analysis local. Only audio goes to the cloud for transcription."

**Q: How accurate is the compliance checking?**
A: "We use RAG with verified regulatory documents, plus guardian review for quality assurance."

**Q: Can it scale to thousands of users?**
A: "Absolutely. The local processing means cloud costs scale linearly, not exponentially."