new_text = """<bos>
Create a high-level plan for completing a household task using the allowed actions. Follow the exact output format described in the examples. Only output the next steps of the plan. Always try to navigate to the object before interacting with it.


Allowed actions: OpenObject, CloseObject, PickupObject, PutObject, ToggleObjectOn, ToggleObjectOff, SliceObject, Navigation

IMPORTANT: Only use objects from this list: AlarmClock, Apple, ArmChair, BaseballBat, BasketBall, Bathtub, BathtubBasin, Bed, Blinds, Book, Boots, Bowl, Box, Bread, ButterKnife, Cabinet, Candle, Cart, CD, CellPhone, Chair, Cloth, CoffeeMachine, CounterTop, CreditCard, Cup, Curtains, Desk, DeskLamp, DishSponge, Drawer, Dresser, Egg, FloorLamp, Footstool, Fork, Fridge, GarbageCan, Glassbottle, HandTowel, HandTowelHolder, HousePlant, Kettle, KeyChain, Knife, Ladle, Laptop, LaundryHamper, LaundryHamperLid, Lettuce, LightSwitch, Microwave, Mirror, Mug, Newspaper, Ottoman, Painting, Pan, PaperTowel, PaperTowelRoll, Pen, Pencil, PepperShaker, Pillow, Plate, Plunger, Poster, Pot, Potato, RemoteControl, Safe, SaltShaker, ScrubBrush, Shelf, ShowerDoor, ShowerGlass, Sink, SinkBasin, SoapBar, SoapBottle, Sofa, Spatula, Spoon, SprayBottle, Statue, StoveBurner, StoveKnob, DiningTable, CoffeeTable, SideTable, TeddyBear, Television, TennisRacket, TissueBox, Toaster, Toilet, ToiletPaper, ToiletPaperHanger, ToiletPaperRoll, Tomato, Towel, TowelHolder, TVStand, Vase, Watch, WateringCan, Window, WineBottle

Task description: pick up a pencil and view it in the light of the lamp
Completed plans: Navigation shelf, PickupObject pencil
Next Plans: Navigation desklamp, ToggleObjectOn desklamp

Task description: Grab a watch and turn a lamp on.
Completed plans: 
Next Plans: Navigation drawer, OpenObject drawer, PickupObject watch, CloseObject drawer, Navigation desklamp, ToggleObjectOn desklamp

Task description: Pick a computer up and turn a lamp on.
Completed plans: Navigation bed, CloseObject laptop, PickupObject laptop, Navigation desklamp
Next Plans: ToggleObjectOn desklamp

Task description: To turn on a lamp and pick up a pen. 
Completed plans: Navigation desklamp, ToggleObjectOn desklamp
Next Plans: PickupObject pen

Task description: examine a pillow with a lamp
Completed plans: Navigation bed, PickupObject pillow, Navigation desklamp
Next Plans: ToggleObjectOn desklamp

Task description: To look at the laptop in the light. 
Completed plans: Navigation sofa, CloseObject laptop, PickupObject laptop
Next Plans: Navigation floorlamp, ToggleObjectOn floorlamp

Task description: examine keys with the lamp
Completed plans: 
Next Plans: Navigation desklamp, ToggleObjectOn desklamp, PickupObject keychain

Task description: Examine a set of keys under a tall lamp
Completed plans: Navigation drawer, OpenObject drawer, PickupObject keychain, CloseObject drawer, Navigation floorlamp
Next Plans: ToggleObjectOn floorlamp

Task description: Examine the baseball bat using the lamp light.
Completed plans: Navigation baseballbat
Next Plans: PickupObject baseballbat, Navigation desklamp, ToggleObjectOn desklamp

Task description: Pick up the cell phone and look at it by the light of the lamp
Completed plans: 
Next Plans: Navigation cellphone, PickupObject cellphone, Navigation desklamp, ToggleObjectOn desklamp



<end_of_turn><eos>"""

print(new_text.split("Next Plans: ")[-1].split("<end_of_turn>")[0].strip())