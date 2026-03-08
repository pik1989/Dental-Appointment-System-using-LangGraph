# Dental Appointment Management System

A conversational AI system for managing dental appointments, powered by LangGraph and Grok-4 (xAI). This project demonstrates a multi-agent architecture where specialized agents work together to handle different appointment-related tasks through natural language interactions.

## Overview

This system provides a chat-based interface for patients and clinic staff to:
- **Check available appointment slots** and doctor information
- **Book new appointments** with preferred doctors
- **Cancel existing appointments**
- **Reschedule appointments** to different time slots

The system uses a supervisor agent that intelligently routes user requests to the appropriate specialized agent based on the detected intent, making it an excellent educational example of multi-agent AI systems.

## Architecture

### Multi-Agent Design

The system follows a supervisor pattern where a central coordinator analyzes user messages and routes them to the most appropriate specialized agent:

```
                    ┌──────────────┐
                    │   Supervisor │ ← Intent classification & routing
                    └──────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
   ┌─────────────┐ ┌─────────────┐ ┌───────────────┐
   │ Info Agent  │ │   Booking   │ │  Cancellation │
   │             │ │    Agent    │ │    Agent      │
   └─────────────┘ └─────────────┘ └───────────────┘
          │
          ▼
   ┌───────────────┐
   │   Reschedule  │
   │    Agent      │
   └───────────────┘
```

### Agent Responsibilities

- **Supervisor**: Analyzes user input, classifies intent (get_info, book, cancel, reschedule, end), and routes to the appropriate agent
- **Info Agent**: Handles queries about available slots, doctor schedules, and patient appointment lookups
- **Booking Agent**: Collects booking details and creates new appointments
- **Cancellation Agent**: Handles appointment cancellation requests
- **Rescheduling Agent**: Manages moving appointments to different time slots

### Technology Stack

- **LangGraph**: Orchestrates the multi-agent workflow and state management
- **LangChain**: Provides the LLM integration and tool framework
- **Grok-4 (xAI)**: Powers the conversational AI capabilities
- **Pandas**: Manages the CSV-based data storage
- **Pydantic**: Handles structured data validation

## Project Structure

```
dental_agent_project/
├── main.py                          # Entry point - interactive CLI
├── doctor_availability.csv          # Data store for appointments
├── requirements.txt                 # Python dependencies
├── dental_agent/
│   ├── agent.py                     # Main agent definition & tools
│   ├── config/
│   │   └── settings.py              # Configuration & environment
│   ├── models/
│   │   └── state.py                 # State schema definitions
│   ├── tools/
│   │   ├── csv_reader.py            # Read operations (query tools)
│   │   └── csv_writer.py            # Write operations (mutation tools)
│   ├── agents/
│   │   ├── supervisor.py            # Intent classification & routing
│   │   ├── info_agent.py            # Information queries
│   │   ├── booking_agent.py         # Appointment booking
│   │   ├── cancellation_agent.py    # Appointment cancellation
│   │   └── rescheduling_agent.py    # Appointment rescheduling
│   └── workflows/
│       └── graph.py                 # LangGraph workflow definition
```

## Installation

### Prerequisites

- Python 3.10 or higher
- An xAI API key (Grok-4 model)

### Steps

1. **Clone the repository** and navigate to the project directory:
   ```bash
   cd dental_agent_project
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:

   Create a `.env` file in the project root with your xAI API key:
   ```
   XAI_API_KEY=your_api_key_here
   MODEL_NAME=grok-4
   TEMPERATURE=0
   ```

   You can obtain an API key from [xAI Console](https://console.x.ai/).

## Usage

### Running the System

Start the interactive appointment management system:

```bash
python main.py
```

### Example Interactions

**Check Available Slots:**
```
You: Show available slots for an orthodontist
Agent: Here are the available orthodontist appointments:
1. 5/10/2026 9:00 - Dr. Emily Johnson
2. 5/10/2026 10:00 - Dr. Emily Johnson
3. 5/12/2026 14:00 - Dr. Emily Johnson
...
```

**Book an Appointment:**
```
You: Book patient 1000082 with Emily Johnson on 5/10/2026 9:00
Agent: Let me check that slot first... The slot is available!
I've booked the appointment:
- Patient ID: 1000082
- Doctor: Emily Johnson
- Date/Time: 5/10/2026 9:00
- Specialization: orthodontist
```

**Check Patient Appointments:**
```
You: What appointments does patient 1000048 have?
Agent: Patient 1000048 has the following appointments:
1. 5/8/2026 9:00 - Dr. John Doe (general_dentist)
```

**Cancel an Appointment:**
```
You: Cancel appointment for patient 1000082 at 5/10/2026 9:00
Agent: I've cancelled the appointment for patient 1000082 on 5/10/2026 at 9:00.
```

**Reschedule an Appointment:**
```
You: Reschedule patient 1000082 from 5/10/2026 9:00 to 5/12/2026 10:00
Agent: Let me verify the new slot is available... It's available!
I've rescheduled the appointment:
- Patient ID: 1000082
- New Date/Time: 5/12/2026 10:00
- Doctor: Emily Johnson
```

## Available Specializations

The system supports the following dental specializations:
- General Dentist
- Oral Surgeon
- Orthodontist
- Cosmetic Dentist
- Prosthodontist
- Pediatric Dentist
- Emergency Dentist

## Data Model

The appointment data is stored in `doctor_availability.csv` with the following structure:

| Field | Description |
|-------|-------------|
| date_slot | Appointment date and time (M/D/YYYY H:MM) |
| specialization | Type of dental specialist |
| doctor_name | Name of the dentist |
| is_available | Boolean indicating if slot is open |
| patient_to_attend | Patient ID if booked, empty if available |

## For Students: How the Flow Works

Understanding this system helps demonstrate several key AI engineering concepts:

### 1. Intent Classification

When a user sends a message, the Supervisor agent analyzes the text to determine what action the user wants. This is done using structured output parsing, where the LLM returns a JSON object with:
- `intent`: The type of request (get_info, book, cancel, reschedule, end)
- `next_agent`: Which specialized agent should handle it
- `reasoning`: Explanation of the decision

### 2. Tool Use in Agents

Each specialized agent has access to specific tools. For example, the Info Agent can query available slots, but cannot book appointments. This demonstrates the principle of least privilege in agent design.

### 3. State Management

LangGraph maintains conversation state across all agents. The state includes:
- Message history (for context)
- Current intent and routing decision
- Parameters collected during booking (patient_id, doctor, date)
- Tool execution results
- Final responses

### 4. Conditional Routing

The graph uses conditional edges to determine flow:
- After supervisor: Route based on classified intent
- After agent: Continue to tools if needed, or end if response is complete

### 5. Data Layer Abstraction

Tools provide an abstraction layer over the CSV data, making it easy to:
- Change the data source (e.g., to a database)
- Add validation
- Modify query logic without changing agent code

## Configuration

Environment variables can be set in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| XAI_API_KEY | Your xAI API key | Required |
| MODEL_NAME | LLM model to use | grok-4 |
| TEMPERATURE | LLM creativity (0=deterministic) | 0 |

## License

This project is provided for educational purposes.
