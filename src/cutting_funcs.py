from funcs.process_event_funcs import *

# Function to apply cuts across all four particles or event-level properties
def apply_cuts(df, cut_list):
    for cut_expression in cut_list:
        quantity, operator, value = parse_cut_expression(cut_expression)
        value = float(value)  # Convert value to float for comparison

        # Check if the quantity is event-level (such as inv_mass_4l)
        if quantity == 'inv_mass_4l':
            # Compute invariant mass of the event
            df['inv_mass_4l'] = inv_mass_4l(df)
            df = df.query(f"inv_mass_4l {operator} {value}")
        elif quantity=='eta':
            for i in range(1, 5):
                cut = f"{quantity}{i} {operator} {value}"
                df = df.query(cut)
        else:
            cut = f"{quantity} {operator} {value}"
            df = df.query(cut)

    return df

def parse_cut_expression(expression):
    for operator in ['<', '>', '<=', '>=', '==']:
        if operator in expression:
            quantity, value = expression.split(operator)
            return quantity.strip(), operator, value.strip()
    raise ValueError(f"Invalid expression: {expression}")

