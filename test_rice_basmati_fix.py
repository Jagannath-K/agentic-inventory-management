from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent

# Create and execute plans to test the fix
print('Testing executor with increased medium priority limit...')
planner = PlannerAgent()
executor = ExecutorAgent()

plans = planner.create_inventory_plan()
medium_plans = [plan for plan in plans if plan.urgency_level == 'MEDIUM']
print(f'Total medium priority plans: {len(medium_plans)}')

# Show first 10 medium priority plans
for i, plan in enumerate(medium_plans[:10], 1):
    print(f'{i}. {plan.product_name}')

print('\nExecuting plans...')
results = executor.process('execute_plans', plans=plans)
print(f'Medium executed: {results["medium_executed"]}')
print(f'Successful orders: {results["successful_orders"]}')

print('\nOrder details for medium priority items:')
for order in results['order_details']:
    if any(plan.product_name == order['product_name'] and plan.urgency_level == 'MEDIUM' for plan in plans):
        status = 'SUCCESS' if order['success'] else 'FAILED'
        print(f'- {order["product_name"]}: {order["order_id"]} - {status}')

# Check specifically for Rice Basmati
rice_basmati_orders = [order for order in results['order_details'] if 'Rice Basmati' in order['product_name']]
if rice_basmati_orders:
    print(f'\n✅ Rice Basmati order found!')
    for order in rice_basmati_orders:
        print(f'   Order ID: {order["order_id"]}, Status: {"SUCCESS" if order["success"] else "FAILED"}')
else:
    print('\n❌ Rice Basmati order NOT found - still being excluded')
