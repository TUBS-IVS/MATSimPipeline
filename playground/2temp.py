# Concating arrays in np is very fast

# passing int or string doesnt matter

# nested dict is much faster than df for most operations

from utils import helpers as h

mydf = h.read_csv("data\\enhanced_frame_final.csv")

# copy columns "km_routing" and "weg_km_imp" to new file
mydf[["km_routing", "wegkm_imp"]].to_csv("data\\km_routing_wegkm_imp.csv", index=False)


