import pandas as pd
import numpy as np

def page_category(df:pd.DataFrame) -> pd.DataFrame:
    df["Page Category"] = df['event_params_page_title'].map(page_dict)
    return df


page_dict = {
    'Home': 'Home',
 'Apparel | Google Merchandise Store': 'Product Listing Page',
 'Frequently Asked Questions': 'Information',
 'Google Online Store': 'Home',
 'Lifestyle': 'Product Listing Page',
 "Men's / Unisex | Apparel | Google Merchandise Store": 'Product Listing Page',
 'Accessories | Google Merchandise Store': 'Product Listing Page',
 'Google Dino Game Tee': 'Product Display Page',
 'Google Tee F/C Black': 'Product Display Page',
 'Campus Collection | Google Merchandise Store': 'Product Listing Page',
 'Socks | Apparel | Google Merchandise Store': 'Product Listing Page',
 'New | Google Merchandise Store': 'New Product Listing Page',
 'Stationery | Google Merchandise Store': 'Product Listing Page',
 'Bags | Lifestyle | Google Merchandise Store': 'Product Listing Page',
 'Shop by Brand | Google Merchandise Store': 'Product Listing Page',
 'Kids | Apparel | Google Merchandise Store': 'Product Listing Page',
 'Store search results': 'Search Results',
 "Men's T-Shirts | Apparel | Google Merchandise Store": 'Product Listing Page',
 'Google Felt Refillable Journal': 'Product Display Page',
 'Payment Method': 'Payment Method',
 'Shopping Cart': 'Shopping Cart',
 'Google Austin Campus Bottle': 'Product Display Page',
 'Sale | Google Merchandise Store': 'Sale Product Listing Page',
 'Checkout Confirmation': 'Checkout Confirmation',
 'Google Tonal Tee Coral': 'Product Display Page',
 'The Google Merchandise Store - Log In': 'Product Listing Page',
 'Google Mural Mug': 'Product Display Page',
 'Checkout Your Information': 'Checkout Your Information',
 'Checkout Review': 'Checkout Review',
 'Google Austin Campus Mug': 'Product Display Page',
 'YouTube | Shop by Brand | Google Merchandise Store': 'Product Listing Page',
 'Google Metallic Notebook Set': 'Product Display Page',
 'Google Sherpa Zip Hoodie Navy': 'Product Display Page',
 'Google Sherpa Vest Black': 'Product Display Page',
 "Google Men's Tech Fleece Grey": 'Product Display Page',
 'Google Sunnyvale Campus Zip Hoodie': 'Product Display Page',
 'BLM Unisex Pullover Hoodie': 'Product Display Page',
 'Google Medium Pet Collar (Blue/Green)': 'Product Display Page',
 'Stickers | Stationery | Google Merchandise Store': 'Product Listing Page',
 'Womens | Apparel | Google Merchandise Store': 'Product Listing Page',
 'Super G Unisex Joggers': 'Product Display Page',
 'Small Goods | Lifestyle | Google Merchandise Store': 'Product Listing Page',
 'Writing | Stationery | Google Merchandise Store': 'Product Listing Page',
 'Drinkware | Lifestyle | Google Merchandise Store': 'Product Listing Page',
 'Hats | Apparel | Google Merchandise Store': 'Product Listing Page',
 'Google Summer19 Crew Grey': 'Product Display Page',
 'Google Crewneck Sweatshirt Navy': 'Product Display Page',
 'Google Campus Bike': 'Product Display Page',
 'Candy Cane Android Cardboard Sculpture': 'Product Display Page',
 'Google Leather Strap Hat Blue': 'Product Display Page',
 'Google Tracking Hat': 'Product Display Page',
 'Google Utility BackPack': 'Product Display Page',
 'Google Land & Sea Nalgene Water Bottle': 'Product Display Page',
 'Google KeepCup': 'Product Display Page',
 'Google Land & Sea Cotton Cap': 'Product Display Page',
 'Android | Shop by Brand | Google Merchandise Store': 'Product Listing Page',
 'Google Unisex Eco Tee Black': 'Product Display Page',
 'Google | Shop by Brand | Google Merchandise Store': 'Product Listing Page',
 'Office | Google Merchandise Store': 'Product Listing Page',
 'The Google Merchandise Store - My Account': 'Product Listing Page',
 "Google Women's Puff Jacket Black": 'Product Display Page',
 'Eco-Friendly | Google Merchandise Store': 'Product Listing Page',
 'Google Womens Microfleece Jacket Black': 'Product Display Page',
 'Notebooks | Stationery | Google Merchandise Store': 'Product Listing Page',
 'Google Incognito Zippack': 'Product Display Page',
 'Return Policy': 'Return Policy',
 'Google Cloth & Pouch Black': 'Product Display Page',
 'Google Infant Charcoal Onesie': 'Product Display Page',
 'Google Toddler Hero Tee Olive': 'Product Display Page',
 'Android Pocket Onesie Navy': 'Product Display Page',
 'Google Badge Heavyweight Pullover Black': 'Product Display Page',
 'Google Land & Sea French Terry Sweatshirt': 'Product Display Page',
 'Google Zip Hoodie F/C': 'Product Display Page',
 'Mugs & Tumblers | Drinkware | Google Merchandise Store': 'Product Listing Page',
 'Google Bot': 'Product Display Page',
 'Google Clear Pen 4-Pack': 'Product Display Page',
 'Google Black Cork Journal': 'Product Display Page',
 'Page Unavailable': 'Other',
 'Google Incognito Techpack V2': 'Product Display Page',
 'Google Beekeepers Tee Mint': 'Product Display Page',
 '#IamRemarkable Unisex Hoodie': 'Product Display Page',
 'Google Seattle Campus Ladies Tee': 'Product Display Page',
 'YouTube Twill Sandwich Cap Black': 'Product Display Page',
 'Water Bottles | Drinkware | Google Merchandise Store': 'Product Listing Page',
 'Google Crew Grey': 'Product Display Page',
 'Shipping Information': 'Shipping Information',
 "Men's T-Shirts": 'Product Listing Page',
 'Google Phone Stand Bamboo': 'Product Display Page',
 'Google Campus Bike Eco Tee Navy': 'Product Display Page',
 'Google Cotopaxi Shell': 'Product Display Page',
 'Google Sherpa Zip Hoodie Charcoal': 'Product Display Page',
 'Android Pocket Toddler Tee White': 'Product Display Page',
 'Android Iconic Sock': 'Product Display Page',
 'Google Leather Strap Hat Black': 'Product Display Page',
 'Google Youth F/C Pullover Hoodie': 'Product Display Page',
 'Google Navy Speckled Tee': 'Product Display Page',
 'Google Infant Hero Tee Olive': 'Product Display Page',
 'Android Garden Tee Orange': 'Product Display Page',
 'Womens Google Striped LS': 'Product Display Page',
 'YouTube Crew Socks': 'Product Display Page',
 'Google Land & Sea Mug': 'Product Display Page',
 'Google Crew Striped Athletic Sock': 'Product Display Page',
 'Google Speckled Beanie Navy': 'Product Display Page',
 'Android Iconic Hat White': 'Product Display Page',
 'Android Iconic Crew': 'Product Display Page',
 'Google Mini Kick Ball': 'Product Display Page',
 '#IamRemarkable Ladies T-Shirt': 'Product Display Page',
 'Android Iconic Hat Green': 'Product Display Page',
 'Google Heather Green Speckled Tee': 'Product Display Page',
 'Google Speckled Beanie Grey': 'Product Display Page',
 'YouTube Leather Strap Hat Black': 'Product Display Page',
 'Google Infant Hero Onesie Grey': 'Product Display Page',
 'Google Youth Hero Tee Grey': 'Product Display Page',
 'Terms of Use': 'Other',
 'Google Land & Sea Journal Set': 'Product Display Page',
 '#IamRemarkable | Shop by Brand | Google Merchandise Store': 'Product Listing Page',
 'Google Crew Combed Cotton Sock': 'Product Display Page',
 'Google Crew Socks': 'Product Display Page',
 "Google Women's Grid Zip-Up": 'Product Display Page',
 'Android Iconic Beanie': 'Product Display Page',
 'Google Toddler FC Zip Hoodie': 'Product Display Page',
 'Google Toddler Tee White': 'Product Display Page',
 'Privacy Policy': 'Other',
 "Google Land & Sea Women's Eco Tee": 'Product Display Page',
 'Google Blue Stojo Cup': 'Product Display Page',
 'Android Pocket Onesie White': 'Product Display Page',
 'Google Land & Sea Tote Bag': 'Product Display Page',
 'Google Land & Sea Unisex Tee': 'Product Display Page',
 'Google Mural Socks': 'Product Display Page',
 'Google Heathered Pom Beanie': 'Product Display Page',
 'Google F/C Long Sleeve Charcoal': 'Product Display Page',
 'YouTube Icon Tee Grey': 'Product Display Page',
 'Google Youth FC Zip Hoodie': 'Product Display Page',
 'Stan and Friends Tee Green': 'Product Display Page',
 'Google Land & Sea Tech Taco': 'Product Display Page',
 'Google Toddler FC Tee Charcoal': 'Product Display Page',
 'Mural Food Container': 'Product Display Page',
 'Android Iconic Hat V.2 Black': 'Product Display Page',
 'Google Mens Microfleece Jacket Black': 'Product Display Page',
 'Google Soft Modal Scarf': 'Product Display Page',
 'Google Chrome Dinosaur Collectible': 'Product Display Page',
 'Essential Canvas Tote': 'Product Display Page',
 "Google Women's Tech Fleece Grey": 'Product Display Page',
 'Electronics | Google Merchandise Store': 'Product Listing Page',
 'Google Campus Bike Tote Navy': 'Product Display Page',
 'Google Packable Bag Black': 'Product Display Page',
 'Google Super G Tumbler (Red Lid)': 'Product Display Page',
 'Google Cork Pencil Pouch': 'Product Display Page',
 'Google Hemp Tote': 'Product Display Page',
 'Android Pocket Youth Tee Green': 'Product Display Page',
 'Google Flat Front Bag Grey': 'Product Display Page',
 'Mommy Works at Google Book': 'Product Display Page',
 'Google Magnet': 'Product Display Page',
 'Google Sustainable Pencil Pouch': 'Product Display Page',
 'Google Youth FC Tee Charcoal': 'Product Display Page',
 'Google Black Tee': 'Product Display Page',
 'Stainless Bent Straw/ Cleaner': 'Product Display Page',
 'Backpacks | Bags | Google Merchandise Store': 'Product Listing Page',
 'Supernatural Paper Backpack': 'Product Display Page',
 'Google LA Campus Mug': 'Product Display Page',
 'Google Glass Bottle': 'Product Display Page',
 'Google Tee Dark Blue': 'Product Display Page',
 'nan': 'Other',
 'Keyboard DOT Sticker': 'Product Display Page',
 'Google Merchandise Store - Forgot Password': 'Other',
 'Android Puzzlebot v2': 'Product Display Page',
 'Google Camp Mug Ivory': 'Product Display Page',
 'Google Pen White': 'Product Display Page',
 'Google Merchandise Store - Reset Password': 'Other',
 "Men's Warm Gear | Apparel | Google Merchandise Store": 'Product Listing Page',
 'Google Raincoat Navy': 'Product Display Page',
 'Google Clear Framed Gray Shades': 'Product Display Page',
 "Google Men's Softshell Moss": 'Product Display Page',
 "Google Men's Discovery Lt. Rain Shell": 'Product Display Page',
 'Google Clear Framed Blue Shades': 'Product Display Page',
 'Google Tee Red': 'Product Display Page',
 'Google F/C Long Sleeve Tee Ash': 'Product Display Page',
 'Google Color Block Notebook': 'Product Display Page',
 'Google Perk Thermal Cup': 'Product Display Page',
 'Google Campus Bike Mini Backpack': 'Product Display Page',
 'Noogler Android Figure 2019': 'Product Display Page',
 'Android SM S/F18 Sticker Sheet': 'Product Display Page',
 'Your Wishlist': 'Product Display Page',
 'Google Boulder Campus Zip Hoodie': 'Product Display Page',
 'Gift Cards | Google Merchandise Store': 'Gift Cards',
 'Google Mountain View Campus Unisex Tee': 'Product Display Page',
 'Google Premium Sunglasses': 'Product Display Page',
 'Google Emoji Sticker Pack': 'Product Display Page',
 'Google Unisex Pride Eco-Tee Black': 'Product Display Page',
 'Google Incognito Dopp Kit V2': 'Product Display Page',
 'Google Striped Penny Pouch': 'Product Display Page',
 'Unisex Google Jumbo Print Tee White': 'Product Display Page',
 'Google Confetti Accessory Pouch': 'Product Display Page',
 'Flamingo and Friends Tee Blue': 'Product Display Page',
 'Google Seaport Tote': 'Product Display Page',
 'Google Woodtop Bottle Black': 'Product Display Page',
 'Google LoveHandle Black': 'Product Display Page',
 'Google Mural Bottle': 'Product Display Page',
 'Google Tee Yellow': 'Product Display Page',
 'Google Toddler Hero Tee Black': 'Product Display Page',
 'Google Tee Green': 'Product Display Page',
 'Google NYC Campus Unisex Tee': 'Product Display Page',
 'Google Cork Base Tumbler': 'Product Display Page',
 'Google Pride Sticker': 'Product Display Page',
 'Google F/C Longsleeve Ash': 'Product Display Page',
 'Google NYC Campus Mug': 'Product Display Page',
 'Google Tee Grey': 'Product Display Page',
 'Google Incognito Flap Pack': 'Product Display Page',
 'Google Tudes Recycled Tee': 'Product Display Page',
 'Google Youth FC Longsleeve Charcoal': 'Product Display Page',
 'Google Crewneck Sweatshirt Green': 'Product Display Page',
 "Google Women's Discovery Lt. Rain Shell": 'Product Display Page',
 'The Google Merchandise Store/Malibu Sunglasses': 'Product Listing Page',
 'YouTube 25 oz Gear Cap Bottle Black': 'Product Display Page',
 'Stan and Friends Toddler Tee Green': 'Product Display Page',
 'The Google Merchandise Store/Maze Pen': 'Product Listing Page',
 'Google Decal': 'Product Display Page',
 'YouTube Icon Tee Charcoal': 'Product Display Page',
 "Google Men's Puff Jacket Black": 'Product Display Page',
 'Google Austin Campus Ladies Tee': 'Product Display Page',
 'Google SF Campus Mug': 'Product Display Page',
 'Google LA Campus Tote': 'Product Display Page',
 "YouTube Women's Favorite Tee White": 'Product Display Page',
 'Shopping & Totes | Bags | Google Merchandise Store': 'Product Listing Page',
 "Google Men's Tech Fleece Vest Charcoal": 'Product Display Page',
 'YouTube Play Mug': 'Product Display Page',
 'Android Iconic Mug Gray': 'Product Display Page',
 'Google Tee Mint Green': 'Product Display Page',
 'Google Youth Badge Tee Navy': 'Product Display Page',
 'Google Youth Hero Tee Maroon': 'Product Display Page',
 'Android Iconic Backpack': 'Product Display Page',
 'Google Stylus Pen w/ LED Light': 'Product Display Page',
 'YouTube Jotter Task Pad': 'Product Display Page',
 "Google Women's Kirkland Pullover": 'Product Display Page',
 'Google Mountain View Campus Bottle': 'Product Display Page',
 'YouTube Small Sticker Sheet': 'Product Display Page',
 'YouTube Standards Zip Hoodie Black': 'Product Display Page',
 'Google Felt Luggage Tag': 'Product Display Page',
 'Stainless Straight Straw/ Cleaner': 'Product Display Page',
 'Google Mountain View Campus Sticker': 'Product Display Page',
 'Black Lives Matter | Google Merchandise Store': 'Product Listing Page',
 'Android Pocket Tee Green': 'Product Display Page',
 "Google Women's Softshell Moss": 'Product Display Page',
 'Google Cup Cap Tumbler Grey': 'Product Display Page',
 'Google Campus Bike Carry Pouch': 'Product Display Page',
 'Google Cork Passport Holder': 'Product Display Page',
 'Supernatural Paper Lunch Sack': 'Product Display Page',
 'Supernatural Paper Tote': 'Product Display Page',
 'Google 16 oz Tumbler Blue': 'Product Display Page',
 'Google Thermal Tumbler Navy': 'Product Display Page',
 'Google 24oz Ring Bottle Blue': 'Product Display Page',
 'Google 24oz Ring Bottle Red': 'Product Display Page',
 'Google Small White Gift Bag 5/PK': 'Product Display Page',
 'Audio | Electronics | Google Merchandise Store': 'Product Listing Page',
 "Google Women's Eco Tee Black": 'Product Display Page',
 'Google Lapel Pin': 'Product Display Page',
 'Google Jotter Task Pad': 'Product Display Page',
 'Android Jotter Task Pad': 'Product Display Page',
 'Youth Jumbo Print Tee White': 'Product Display Page',
 'Google Youth Badge Tee Olive': 'Product Display Page',
 'Google Kids Playful Tee': 'Product Display Page',
 'Daddy Works at Google Book': 'Product Display Page',
 'Google SF Campus Lapel Pin': 'Product Display Page',
 'Google Confetti Task Pad': 'Product Display Page',
 'Android Pocket Toddler Tee Navy': 'Product Display Page',
 'Google Incognito Messenger Bag': 'Product Display Page',
 'Google Clear Framed Yellow Shades': 'Product Display Page',
 'Android Super Hero 3D Framed Art': 'Product Display Page',
 'Google Utensil Set': 'Product Display Page',
 'Unisex Google Pocket Tee Grey': 'Product Display Page',
 'Google See-No Hear-No Set': 'Product Display Page',
 'Google Laptop Sleeve Charcoal': 'Product Display Page',
 'Google Austin Campus Unisex Tee': 'Product Display Page',
 'Google Tonal Tee Spearmint': 'Product Display Page',
 'Google Mountain View Campus Zip Hoodie': 'Product Display Page',
 'Google NYC Campus Zip Hoodie': 'Product Display Page',
 '#IamRemarkable Tote': 'Product Display Page',
 '#IamRemarkable Unisex T-Shirt': 'Product Display Page',
 'Google NYC Campus Ladies Tee': 'Product Display Page',
 'Google Mountain View Tee Blue': 'Product Display Page',
 'Google Felt Mason Jar': 'Product Display Page',
 'Google Green YoYo': 'Product Display Page',
 'Google Blue YoYo': 'Product Display Page',
 'Google LA Campus Unisex Tee': 'Product Display Page',
 'YouTube Transmission Journal Red': 'Product Display Page',
 'YouTube Iconic Play Pin': 'Product Display Page',
 'Google Large Tote White': 'Product Display Page',
 'Google Tee Blue': 'Product Display Page',
 'Google Split Seam Tee Olive': 'Product Display Page',
 'Google LA Campus Ladies Tee': 'Product Display Page',
 "Google Women's Black Tee": 'Product Display Page',
 'Google Large Pet Collar (Blue/Green)': 'Product Display Page',
 'Android Buoy Bottle': 'Product Display Page',
 'Google Mural Sticky Note Pad': 'Product Display Page',
 'Google Sunnyvale Campus Unisex Tee': 'Product Display Page',
 '#IamRemarkable Lapel Pin': 'Product Display Page',
 'Google Red Speckled Tee': 'Product Display Page',
 'Google Chicago Campus Mug': 'Product Display Page',
 'Google Confetti Slim Task Pad': 'Product Display Page',
 'Google SF Campus Zip Hoodie': 'Product Display Page',
 'Google Totepak': 'Product Display Page',
 'Google Tech Taco': 'Product Display Page',
 'Google Small Cable Organizer Blue': 'Product Display Page',
 '(direct)': 'Other',
 'Android Iconic Glass Bottle Green': 'Product Display Page',
 'Google Keychain': 'Product Display Page',
 'Google Red Kids Sunglasses': 'Product Display Page',
 'Google Cork Journal': 'Product Display Page',
 'Google Medium Pet Collar (Red/Yellow)': 'Product Display Page',
 'Google Medium Pet Leash (Blue/Green)': 'Product Display Page',
 'Google Cambridge Campus Zip Hoodie': 'Product Display Page',
 "Google Women's Tech Fleece Vest Charcoal": 'Product Display Page',
 'Google Incognito Laptop Organizer': 'Product Display Page',
 'Google Recycled Notebook Set Natural': 'Product Display Page',
 'Google Small Standard Journal Navy': 'Product Display Page',
 'Android Large Trace Journal Black': 'Product Display Page',
 'Google Large Standard Journal Grey': 'Product Display Page',
 'Sale | Sale-Accessories': 'Sale Product Listing Page',
 "Google Women's Pride Eco-Tee Black": 'Product Display Page',
 'Google Mesh Bag Blue': 'Product Display Page',
 'Android Large Removable Sticker Sheet': 'Product Display Page',
 "Women's Warm Gear | Apparel | Google Merchandise Store": 'Product Listing Page',
 'Google Super G Tumbler (Blue Lid)': 'Product Display Page',
 "Women's T-Shirts | Apparel | Google Merchandise Store": 'Product Listing Page',
 'Google NYC Campus Lapel Pin': 'Product Display Page',
 'Google SF Campus Unisex Tee': 'Product Display Page',
 'Google Cambridge Campus Ladies Tee': 'Product Display Page',
 'Google Mural Sticker Sheet': 'Product Display Page',
 'Google SF Campus Tote': 'Product Display Page',
 'Google Light Pen Red': 'Product Display Page',
 'Google Pen Lilac': 'Product Display Page',
 'Android Techie 3D Framed Art': 'Product Display Page',
 'Google SF Campus Ladies Tee': 'Product Display Page',
 'Google Seattle Campus Sticker': 'Product Display Page',
 'Android Pocket Tee Navy': 'Product Display Page',
 'Google Laptop and Cell Phone Stickers': 'Product Display Page',
 'Google Canteen Bottle Black': 'Product Display Page',
 'Google Light Pen Green': 'Product Display Page',
 'Google Pen Red': 'Product Display Page',
 '#IamRemarkable Water Bottle': 'Product Display Page',
 'Google Confetti Pen White': 'Product Display Page',
 'Google Campus Bike Bottle': 'Product Display Page',
 'The Google Merchandise Store - Register': 'Product Listing Page',
 'Google Mouse Pad Navy': 'Product Display Page',
 'Google Maps Pin': 'Product Display Page',
 'Android Iconic Pin': 'Product Display Page',
 'Google Cambridge Campus Mug': 'Product Display Page',
 'Google Cork Card Holder': 'Product Display Page',
 'Android Hipster Pin': 'Product Display Page',
 'Google Cambridge Campus Tote': 'Product Display Page',
 'Google Chicago Campus Tote': 'Product Display Page',
 'Stan and Friends Onesie Green': 'Product Display Page',
 'Google Large White Gift Bag 5/PK': 'Product Display Page',
 'Google Sunnyvale Campus Tote': 'Product Display Page',
 'Google Chicago Campus Zip Hoodie': 'Product Display Page',
 "Infant | Kids' Apparel | Google Merchandise Store": 'Product Listing Page',
 'Google Austin Campus Zip Hoodie': 'Product Display Page',
 'Google Seattle Campus Unisex Tee': 'Product Display Page',
 'Google Mesh Bag Red': 'Product Display Page',
 'Google Tudes Thermal Bottle': 'Product Display Page',
 'Google ApPeel Journal Red': 'Product Display Page',
 'Google Frisbee': 'Product Display Page',
 'Android Small Trace Journal Black': 'Product Display Page',
 'Android Iconic Notebook': 'Product Display Page',
 'Google Chicago Campus Lapel Pin': 'Product Display Page',
 'Google Emoji Magnet Set': 'Product Display Page',
 'Google Utility Bag Grey': 'Product Display Page',
 'Google Sunnyvale Campus Sticker': 'Product Display Page',
 'Google Campus Bike Corkbase Mug Blue': 'Product Display Page',
 'Google PNW Campus Zip Hoodie': 'Product Display Page',
 'Google LA Campus Zip Hoodie': 'Product Display Page',
 'Google Light Pen Blue': 'Product Display Page',
 'Google LA Campus Lapel Pin': 'Product Display Page',
 'Android Iconic Pen': 'Product Display Page',
 'Google Blue Kids Sunglasses': 'Product Display Page',
 'Google Seattle Campus Mug': 'Product Display Page',
 'Google Felt Strap Keyring': 'Product Display Page',
 'Google Recycled Writing Set': 'Product Display Page',
 'Google Chicago Campus Unisex Tee': 'Product Display Page',
 'Google Pen Grass Green': 'Product Display Page',
 'หน้าไม่พร้อมใช้งาน': 'Product Display Page',
 'Google NYC Campus Sticker': 'Product Display Page',
 'Google Mural Tote': 'Product Display Page',
 'Google Beekeepers Onesie Pink': 'Product Display Page',
 'Google Cork Key Ring': 'Product Display Page',
 'Stan and Friends Youth Tee Green': 'Product Display Page',
 'Google Mountain View Campus Tote': 'Product Display Page',
 'Google Chicago Campus Bottle': 'Product Display Page',
 'Google Sunnyvale Campus Bottle': 'Product Display Page',
 'More Bags | Bags | Google Merchandise Store': 'Product Listing Page',
 'Android Pocket Youth Tee Navy': 'Product Display Page',
 'Google PNW Campus Ladies Tee': 'Product Display Page',
 'Google Kirkland Campus Unisex Tee': 'Product Display Page',
 '#IamRemarkable Journal': 'Product Display Page',
 'Android Iconic 4in Decal': 'Product Display Page',
 'Google NYC Campus Bottle': 'Product Display Page',
 "Youth | Kids' Apparel | Google Merchandise Store": 'Product Listing Page',
 'Google Sunnyvale Campus Ladies Tee': 'Product Display Page',
 'Google Recycled Pen Black': 'Product Display Page',
 'Google Red YoYo': 'Product Display Page',
 'Google Yellow YoYo': 'Product Display Page',
 'Google Cork Tablet Case': 'Product Display Page',
 'Google Separating Keyring': 'Product Display Page',
 'Android Geek Pin': 'Product Display Page',
 'Google Boulder Campus Mug': 'Product Display Page',
 'Google PNW Campus Unisex Tee': 'Product Display Page',
 'TYCTWD | Google Merchandise Store': 'Product Listing Page',
 'Fun | Accessories | Google Merchandise Store': 'Product Listing Page',
 'Snowflake Android Cardboard Sculpture': 'Product Display Page',
 'Google Campus Bike Grid Task Pad': 'Product Display Page',
 'Google Kirkland Campus Mug': 'Product Display Page',
 'Google Austin Campus Lapel Pin': 'Product Display Page',
 'Google Mural Notebook': 'Product Display Page',
 'Google Recycled Pen Green': 'Product Display Page',
 'Google Cambridge Campus Lapel Pin': 'Product Display Page',
 'Google Cambridge Campus Bottle': 'Product Display Page',
 'Google Boulder Campus Unisex Tee': 'Product Display Page',
 "Toddler | Kids' Apparel | Google Merchandise Store": 'Product Listing Page',
 'Google Cambridge Campus Unisex Tee': 'Product Display Page',
 'Google Beekeepers Toddler Tee Pink': 'Product Display Page',
 'Google Beekeepers Youth Tee Pink': 'Product Display Page',
 'Office |Miscellaneous | Google Merchandise Store': 'Product Listing Page',
 'Google PNW Campus Mug': 'Product Display Page',
 'Google Bottle Cleaner': 'Product Display Page',
 'Google Bear Baby Blanket Beige': 'Product Display Page',
 'Google Large Pet Leash (Red/Yellow)': 'Product Display Page',
 'Google Mountain View Tee Red': 'Product Display Page',
 'Google Pen Citron': 'Product Display Page',
 'Google Pen Bright Blue': 'Product Display Page',
 'Android Lumberjack Pin': 'Product Display Page',
 'Android Iconic Sticker Sheet': 'Product Display Page',
 'Google PNW Campus Sticker': 'Product Display Page',
 'Google PNW Campus Tote': 'Product Display Page',
 'Google Large Pet Leash (Blue/Green)': 'Product Display Page',
 '#IamRemarkable Pen': 'Product Display Page',
 'Google Boulder Campus Bottle': 'Product Display Page',
 'Google Boulder Campus Ladies Tee': 'Product Display Page',
 'Google Pen Grey': 'Product Display Page',
 'Google Pen Neon Coral': 'Product Display Page',
 'Page non disponible': 'Product Display Page',
 'Google Mountain View Campus Ladies Tee': 'Product Display Page',
 'Google Austin Campus Tote': 'Product Display Page',
 'Google Sunnyvale Campus Mug': 'Product Display Page',
 'The Google Merchandise Store/Gift Card - $25.00': 'Gift Cards',
 'Google LA Campus Sticker': 'Product Display Page',
 'Google SF Campus Bottle': 'Product Display Page',
 'Google Boulder Campus Sticker': 'Product Display Page',
 'Google Kirkland Campus Sticker': 'Product Display Page',
 'Google Bellevue Campus Sticker': 'Product Display Page',
 'Google Cambridge Campus Sticker': 'Product Display Page',
 'Sale | Sale-Apparel': 'Sale Product Listing Page',
 'Clearance Sale': 'Sale Product Listing Page',
 'The Google Merchandise Store/Gift Card - $50.00': 'Gift Cards',
 'Google Crewneck Sweatshirt Grey': 'Product Display Page',
 'Google Grey Tee': 'Product Display Page',
 'New | New-Accessories': 'New Product Listing Page',
 'Página no disponible': 'Product Display Page',
 'Waze | Google Merchandise Store': 'Product Listing Page',
 'Google Large Pet Collar (Red/Yellow)': 'Product Display Page',
 'Google Chicago Campus Ladies Tee': 'Product Display Page',
 'The Google Merchandise Store/Gift Card - $10.00': 'Gift Cards',
 'Accessories | Electronic Accessories | Google Merchandise Store': 'Product Listing Page',
 'Google Kirkland Campus Lapel Pin': 'Product Display Page',
 'Google Seattle Campus Lapel Pin': 'Product Display Page',
 'Google F/C Longsleeve Charcoal': 'Product Display Page'
 }
