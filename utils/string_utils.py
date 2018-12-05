
def pretty_print_table(table):
        # table in format: [(t_1, ..., t_n)]
        
        # 0) Expand the cells
        table = expand_cells(table)
        
        # 1) Every row should have equal height
        table = equalize_row_heights(table)
        
        # 2) Every column should be same width
        table = equalize_column_widths(table)
        
        # 3) Merge the columns together as a single string
        # [[[]...[]]...[[]...[]]] ==> [rows]
        table = merge_columns(table)
        
        # 4) Merge rows toghether ad a single string
        table = merge_rows(table)
        
        return table
        
        
def table_to_columns(table):
    # Empty table returns empty table
    if len(table) == 0:
        return []
    
    # Get the columns and put them in a list
    lists = []
    n_l = len(table[0])
    for i in range(n_l):
        lists.append(select_column(table, i))
        
    return lists

def select_column(table, column):
    return [row[column] for row in table]
        
def space_out_strings(strings):
    strings = list(map(str, strings))                           # To string
    max_l = max(list(map(len, strings)))                        # Length
    return [space_out_string(x, max_l) for x in strings]        # Add spaces if needed

def space_out_string(string, max_l):
    return string + (" " * (max_l - len(string)))
    
def expand_cells(table):
    # table: [(t1,...,tN)]
    
    # 1) Every multi-line cell needs to be split
    # [(t1,...,tn)] ==> [[[l1,...,lm]1, ..., [l1,...,lm]n]]
    return [expand_row(row) for row in table]    
    
def expand_row(row):
    # Row: (t1, ..., tn)  
    
    row = [str(cell) for cell in row]  
    return [cell.split("\n") for cell in row]
    
def equalize_row_heights(table):
    #table: [[[]...[]], ..., [[],...,[]]]
    # 0: row
    # 1: column
    # 2: cell part
    
    return [equalize_row_height(row) for row in table]
    
def equalize_row_height(row):
    # row: [[], ..., []]
    #  0: column
    #  1: cell part
    
    heights = [get_cell_height(cell) for cell in row]
    max_height = max(heights)
    
    return [resize_cell_height(cell, max_height) for cell in row]
    
def resize_cell_height(cell, max_height):
    # cell: []
    # 0: cell part
    
    cell_height = get_cell_height(cell)
    height_to_add = max_height - cell_height
    
    return cell + ([""]*height_to_add)
    
def resize_cell_width(cell, max_width):
    # cell: []
    # 0: cell part
    
    return [resize_string(cell_part, max_width) for cell_part in cell]
    
def resize_string(string, max_width):
    string_width = len(string)
    width_to_add = max_width - string_width
    
    return string + (" " * width_to_add)
    
def get_cell_height(cell):
    return len(cell)
    
def get_cell_width(cell):
    return max([len(cell_part) for cell_part in cell])
    
def equalize_column_widths(table):
    #table: [[[]...[]], ..., [[],...,[]]]
    # 0: row
    # 1: column
    # 2: cell part
    
    # Convert to columns
    columns = table_to_columns(table)
    columns = [equalize_column_width(column) for column in columns]
    
    return list(zip(*columns))
    
def equalize_column_width(column):
    # column: [[],...,[]]
    # 0: row
    # 1: cell part
    
    cell_widths = [get_cell_width(cell) for cell in column]
    max_width = max(cell_widths)
    
    return [resize_cell_width(cell, max_width) for cell in column]
    
    
def merge_columns(table):
    #table: [[[]...[]], ..., [[],...,[]]]
    # 0: row
    # 1: column
    # 2: cell part
    
    return [merge_column(row) for row in table]
    
def merge_column(row):
    # row: [[]...[]]
    
    n_columns = len(row)
    if n_columns == 0:
        return []
        
    n_parts = len(row[0])
    merged_parts = [merge_part(row, part) for part in range(n_parts)]
    
    return "\n".join(merged_parts)
    
def merge_part(row, part):
    parts = [cell[part] for cell in row]
    return " ".join(parts)
    
def merge_rows(rows):

    # Merging nothing results in nothing
    n_rows = len(rows)
    if n_rows == 0:
        return []
    
    table_width = len(rows[0])
    delim = "\n" + "" * table_width
    return delim.join(rows)
    
    
