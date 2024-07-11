from dd.autoref import BDD

bdd = BDD()
bdd.declare("col_y", "col_w", "tag_o", "tag_b")
kb = bdd.add_expr(r"(col_y&tag_o)|(col_w&tag_b)")

rule1 = bdd.add_expr(r"col_y=>(tag_o&!tag_b)")
rule2 = bdd.add_expr(r"col_w=>(tag_b&!tag_o)")
kb2 = bdd.apply("and", rule1, rule2)

anorm = bdd.add_expr(r"col_y&tag_b")
print(bdd.apply("and", kb2, anorm).to_expr())

# MDD
import mdd
