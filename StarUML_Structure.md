# StarUML Diagram Structure for Agentic Inventory Management System

## Project Overview
**System Name**: Agentic Inventory Management System
**Architecture**: Multi-Agent AI System with Web Interface
**Technology Stack**: Python, Streamlit, Pandas, Scikit-learn, SMTP

---

## 1. Package Structure

### Main Packages:
```
📦 AgenticInventorySystem
├── 📁 agents/           # AI Agent Components
├── 📁 models/           # Data Models & ML Components  
├── 📁 ui/               # User Interface Components
├── 📁 data/             # Data Storage & Management
└── 📁 config/           # Configuration Management
```

---

## 2. Class Diagram Structure

### Package: agents/

#### Abstract Base Class
```uml
<<abstract>>
BaseAgent
- agent_name: str
- logger: Logger
+ __init__(name: str)
+ process(action: str, **kwargs): Any
+ log_activity(message: str): void
```

#### Concrete Agent Classes
```uml
PlannerAgent extends BaseAgent
- sales_data: DataFrame
- stock_data: DataFrame
- supplier_data: DataFrame
+ load_data(): void
+ analyze_demand_patterns(product_id: str): dict
+ calculate_optimal_reorder_point(product_id: str): tuple
+ forecast_demand(product_id: str, days: int): float
+ calculate_optimal_order_quantity(product_id: str): int
+ create_inventory_plan(): List[InventoryPlan]
+ process(action: str, **kwargs): Any

ExecutorAgent extends BaseAgent
- stock_data: DataFrame
- supplier_data: DataFrame
- pending_orders: List[OrderRequest]
- completed_orders: List[OrderResult]
- order_counter: int
- demand_predictor: DemandPredictor
- notification_system: NotificationSystem
- supplier_email: str
+ load_data(): void
+ create_order_request(plan: InventoryPlan): OrderRequest
+ validate_order_request(request: OrderRequest): dict
+ simulate_order_placement(request: OrderRequest): OrderResult
+ send_order_email(request: OrderRequest, order_id: str): bool
+ update_inventory_records(request: OrderRequest, result: OrderResult): bool
+ execute_inventory_plans(plans: List[InventoryPlan]): dict
+ process(action: str, **kwargs): Any

ReflectorAgent extends BaseAgent
- performance_data: dict
- analysis_results: dict
+ load_data(): void
+ analyze_system_performance(): dict
+ generate_insights(): List[dict]
+ create_recommendations(): List[str]
+ assess_stockout_risks(): dict
+ process(action: str, **kwargs): Any
```

### Package: models/

#### Data Transfer Objects
```uml
<<dataclass>>
InventoryPlan
+ product_id: str
+ product_name: str
+ current_stock: int
+ predicted_demand: float
+ reorder_quantity: int
+ reorder_date: datetime
+ urgency_level: str
+ reasoning: str
+ confidence_score: float

<<dataclass>>
OrderRequest
+ product_id: str
+ product_name: str
+ quantity: int
+ supplier_id: str
+ estimated_cost: float
+ priority: str
+ requested_date: datetime
+ expected_delivery: datetime
+ order_id: Optional[str]
+ status: str

<<dataclass>>
OrderResult
+ order_id: str
+ success: bool
+ message: str
+ estimated_delivery: Optional[datetime]
+ actual_cost: Optional[float]
```

#### Business Logic Classes
```uml
ShopOperations
- config: dict
- notification_system: NotificationSystem
- transaction_log: str
+ __init__(config: dict)
+ load_data(): tuple
+ check_stock_levels(): List[dict]
+ process_transactions(transactions: List[dict]): List[dict]
+ _send_stock_alerts(alerts: List[dict]): void
+ generate_reports(): dict
+ update_stock(product_id: str, quantity: int): bool

DemandPredictor
- models: dict
- scalers: dict
- trained_models: dict
- feature_importance: dict
- model_performance: dict
- sales_data: DataFrame
- stock_data: DataFrame
+ load_data(): void
+ create_features(sales_data: DataFrame, product_id: str): DataFrame
+ train_models(product_id: str): dict
+ predict_demand(product_id: str, target_date: datetime): dict
+ evaluate_models(product_id: str, X_test: array, y_test: array): dict

NotificationSystem
- email_host: str
- email_port: int
- email_user: str
- email_password: str
+ __init__()
+ send_stock_alert(item: dict, recipients: List[str], subject: str): bool
+ send_order_notification(order: dict, recipients: List[str]): bool
+ _create_stock_alert_body(item: dict): str
+ _send_email(recipient: str, subject: str, body: str): bool
```

### Package: ui/

#### User Interface Classes
```uml
StreamlitDashboard
+ load_data(): tuple
+ initialize_agents(): bool
+ create_overview_dashboard(): void
+ create_ai_planning_dashboard(): void
+ create_analytics_dashboard(): void
+ create_demand_forecasting_dashboard(): void
+ main(): void

<<static>>
UIComponents
+ create_metric_card(title: str, value: str, delta: str): void
+ create_status_indicator(status: str): void
+ create_data_table(data: DataFrame, columns: List[str]): void
+ create_chart(data: DataFrame, chart_type: str): void
```

### Package: data/

#### Data Access Layer
```uml
DataManager
- data_path: str
+ load_sales_data(): DataFrame
+ load_stock_data(): DataFrame  
+ load_supplier_data(): DataFrame
+ save_stock_data(data: DataFrame): bool
+ save_transaction_log(transactions: List[dict]): bool
+ backup_data(): bool
+ validate_data_integrity(): bool

ConfigManager
- config_file: str
- config_data: dict
+ load_config(): dict
+ save_config(config: dict): bool
+ get_setting(key: str): Any
+ update_setting(key: str, value: Any): bool
```

---

## 3. Sequence Diagrams

### Use Case 1: Generate Inventory Plan
```sequence
User -> StreamlitDashboard: Click "Generate New Plan"
StreamlitDashboard -> PlannerAgent: create_inventory_plan()
PlannerAgent -> DataManager: load_sales_data()
PlannerAgent -> DataManager: load_stock_data()
PlannerAgent -> PlannerAgent: analyze_demand_patterns()
PlannerAgent -> PlannerAgent: calculate_optimal_reorder_point()
PlannerAgent -> PlannerAgent: forecast_demand()
PlannerAgent -> StreamlitDashboard: return List[InventoryPlan]
StreamlitDashboard -> User: Display plans in table
```

### Use Case 2: Execute Orders
```sequence
User -> StreamlitDashboard: Click "Execute Orders"
StreamlitDashboard -> ExecutorAgent: execute_inventory_plans(plans)
ExecutorAgent -> DemandPredictor: predict_demand()
ExecutorAgent -> ExecutorAgent: create_order_request()
ExecutorAgent -> ExecutorAgent: validate_order_request()
ExecutorAgent -> ExecutorAgent: send_order_email()
ExecutorAgent -> NotificationSystem: send email
ExecutorAgent -> DataManager: update_inventory_records()
ExecutorAgent -> StreamlitDashboard: return execution_results
StreamlitDashboard -> User: Display execution results
```

### Use Case 3: System Analytics
```sequence
User -> StreamlitDashboard: Navigate to Analytics
StreamlitDashboard -> ReflectorAgent: analyze_system_performance()
ReflectorAgent -> DataManager: load_performance_data()
ReflectorAgent -> ReflectorAgent: generate_insights()
ReflectorAgent -> ReflectorAgent: create_recommendations()
ReflectorAgent -> StreamlitDashboard: return analysis_report
StreamlitDashboard -> User: Display analytics dashboard
```

---

## 4. Component Diagram

### System Architecture
```component
[Web Browser] --HTTP--> [Streamlit Server]
[Streamlit Server] --> [UI Components]
[UI Components] --> [Agent Layer]
[Agent Layer] --> [Business Logic Layer]
[Business Logic Layer] --> [Data Access Layer]
[Data Access Layer] --> [CSV Files]
[Agent Layer] --> [ML Models]
[Agent Layer] --> [Email Service]
```

### Dependencies
```
UI Layer depends on:
- agents.planner.PlannerAgent
- agents.executor.ExecutorAgent  
- agents.reflector.ReflectorAgent
- models.predictor.DemandPredictor

Agent Layer depends on:
- models.shop_operations.ShopOperations
- models.notification_system.NotificationSystem
- models.predictor.DemandPredictor

Model Layer depends on:
- pandas, numpy, sklearn
- smtplib, email libraries
- datetime, logging
```

---

## 5. State Diagram

### Order Processing States
```state
[Initial] --> [Planning]
[Planning] --> [Validation] : plan created
[Validation] --> [Execution] : validation passed
[Validation] --> [Planning] : validation failed
[Execution] --> [Email Sent] : email successful
[Execution] --> [Failed] : email failed
[Email Sent] --> [Inventory Updated] : update successful
[Email Sent] --> [Failed] : update failed
[Inventory Updated] --> [Completed]
[Failed] --> [Planning] : retry
```

### System Status States
```state
[Offline] --> [Initializing] : system start
[Initializing] --> [Loading Data] : agents created
[Loading Data] --> [Ready] : data loaded
[Ready] --> [Processing] : user action
[Processing] --> [Ready] : action completed
[Processing] --> [Error] : exception occurred
[Error] --> [Ready] : error handled
[Ready] --> [Offline] : system shutdown
```

---

## 6. Activity Diagram

### Daily Operations Flow
```activity
Start --> Load System Data
Load System Data --> Initialize AI Agents
Initialize AI Agents --> Monitor Stock Levels
Monitor Stock Levels --> {Below Reorder Point?}
{Below Reorder Point?} --Yes--> Generate Inventory Plan
{Below Reorder Point?} --No--> Continue Monitoring
Generate Inventory Plan --> Predict Demand
Predict Demand --> Calculate Order Quantities
Calculate Order Quantities --> Validate Orders
Validate Orders --> {Valid Orders?}
{Valid Orders?} --Yes--> Send Order Emails
{Valid Orders?} --No--> Log Error
Send Order Emails --> Update Inventory Records
Update Inventory Records --> Log Transaction
Log Transaction --> Continue Monitoring
Log Error --> Continue Monitoring
```

---

## 7. Deployment Diagram

### System Deployment
```deployment
<<device>> User Computer
- Web Browser
- Network Connection

<<server>> Application Server
- Python Runtime
- Streamlit Application
- AI Agents
- ML Models

<<database>> Data Storage
- CSV Files (sales.csv, stock.csv, suppliers.csv)
- Log Files
- Model Files (.pkl)

<<service>> Email Service
- Gmail SMTP Server
- Authentication

<<network>> Internet
- HTTPS Connection
- SMTP Connection
```

---

## 8. Package Dependencies

```dependencies
ui.app depends on:
- agents.planner
- agents.executor  
- agents.reflector
- models.predictor

agents.* depends on:
- models.shop_operations
- models.notification_system
- pandas, numpy

models.predictor depends on:
- sklearn.ensemble
- sklearn.linear_model
- sklearn.preprocessing

models.notification_system depends on:
- smtplib
- email.mime
```

---

## 9. Design Patterns Used

1. **Strategy Pattern**: Different ML models in DemandPredictor
2. **Observer Pattern**: NotificationSystem for alerts
3. **Command Pattern**: Agent.process() methods
4. **Factory Pattern**: Agent initialization
5. **Singleton Pattern**: ConfigManager
6. **Template Method**: BaseAgent abstract class
7. **Adapter Pattern**: Data format conversions

---

## 10. Quality Attributes

- **Scalability**: Modular agent architecture
- **Maintainability**: Separation of concerns
- **Reliability**: Error handling and logging
- **Performance**: Caching and optimized ML models
- **Security**: Email authentication and input validation
- **Usability**: Intuitive web interface
- **Extensibility**: Plugin-based agent system

This structure provides a comprehensive view of the system architecture suitable for creating detailed StarUML diagrams with all necessary classes, relationships, and behavioral models.
