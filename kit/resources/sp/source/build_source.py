"""Deterministically render the authored SP source; no models or external data."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
# Each line is task content | hand-written witness | primary argument (if needed).
# ~ is a line break; ^ is a blank line. Neither appears in task content.
DATA = {
    'required_word': '''Suggest a spot for a spare umbrella near the entrance.|Keep the spare umbrella in a basket beside the door.|basket
Offer a way to keep a paperback open while taking notes.|Rest a bookmark across the page you are reading.|bookmark
Recommend a place to collect loose buttons from laundry.|A small jar on the laundry shelf keeps buttons together.|jar
Describe a simple setup for drawing at the kitchen table.|Put a pencil and plain paper on a washable mat.|pencil
Suggest a reminder for returning borrowed garden tools.|Place a reminder beside the tools before you leave.|reminder
Offer a way to separate clean dishcloths from used ones.|Reserve a drawer for clean dishcloths and a hook for used ones.|drawer
Help someone remember which seedlings were planted first.|Add a dated label to each seedling pot.|label
Suggest a quiet activity while waiting for a cake to cool.|Read a magazine while the cake rests on its rack.|magazine
Explain how to make a reusable shopping bag easy to find.|Hang the bag on a hook beside your keys.|hook
Recommend a simple welcome gesture for a new neighbor.|Leave a friendly note introducing yourself.|note
Give a quick tip for arranging a small sewing area.|Keep thread in a shallow tray beside the fabric.|thread
Suggest an item to bring when sketching outdoors.|Pack a cushion so you can sit comfortably while sketching.|cushion
Explain a way to sort reusable food containers.|Group each lid beside its matching container on the shelf.|lid
Recommend a way to mark a favorite soup recipe.|Slip a ribbon between the pages at the soup recipe.|ribbon
Suggest something that makes a balcony breakfast pleasant.|Set a flower on the table beside your breakfast.|flower
Offer an easy way to organize postcards from friends.|Place the postcards in an album with room for notes.|album
Describe a tidy place for wrapping supplies.|Keep paper and tape together in a long storage box.|tape
Propose a small improvement to a shared reading corner.|Add a lamp beside the chair for evening reading.|lamp
Suggest a way to transport a homemade pie to a picnic.|Place the pie in a sturdy carrier with a flat base.|carrier
Explain how to make a watering schedule visible.|Put a calendar beside the watering can and mark each day.|calendar''',
    'forbidden_word': '''Suggest a relaxing activity for a rainy afternoon.|Listen to soft music while sorting a small puzzle.|cozy
Describe a tidy way to store scarves.|Loop them over a hanger so each one stays visible.|closet
Offer a friendly greeting for a walking group.|Glad you could join our walk today.|welcome
Recommend a simple snack for a train ride.|Pack crackers in a small reusable box.|delicious
Give advice for choosing a seat at a community talk.|Pick a place where you can see the speaker clearly.|front
Suggest an indoor activity for a windy weekend.|Try folding paper boats at the kitchen table.|craft
Explain a quick way to freshen a desk arrangement.|Move the lamp aside and clear a space for writing.|clean
Propose a pleasant break between household chores.|Sit by the window and enjoy a warm drink.|tea
Suggest how to display a child's drawing.|Attach it to a cork board with bright pins.|fridge
Offer a way to remember a library due date.|Add the return day to your calendar.|book
Explain how to keep a hallway clear of footwear.|Place a low rack beside the entrance. Leave the walking route open.|shoes
Suggest a small activity for a family reunion.|Bring an old photo album. Invite everyone to share a memory.|game
Describe an easy breakfast for a slow morning.|Toast a slice of bread. Add sliced fruit on the side.|cereal
Recommend a way to organize a reusable lunch bag.|Put the main container at the bottom. Tuck the napkin beside it.|sandwich
Explain how to prepare for a short local bike ride.|Check the tires. Bring a filled water bottle. Choose a familiar route.|helmet
Offer a way to make an afternoon at home feel special.|Arrange a few flowers. Play a favorite album.|luxury
Suggest a polite reminder about shared kitchen space.|Please put your cups away. Leave the counter ready for the next person.|mess
Describe a way to organize loose recipe cards.|Sort the cards by meal. Hold each group with a clip.|box
Recommend an activity for a picnic while food is being unpacked.|Look for interesting leaf shapes. Compare what you find.|ball
Help someone plan a relaxed visit to a local park.|Choose a shaded route. Leave time to sit near the pond.|busy''',
    'exact_word_frequency': '''Describe two ways to use a spare basket at home.|A basket can hold knitting supplies. Another basket can collect scarves.|basket
Explain how a desk lamp helps with different hobbies.|A lamp lights your sketchbook. The same lamp helps you sort beads.|lamp
Describe a simple routine for caring for a houseplant.|Check the plant each morning. Turn the plant toward the light.|plant
Offer ideas for using two empty jars.|One jar can store buttons. Another jar can hold pencils.|jar
Describe a tidy arrangement for a small writing desk.|Keep a notebook beside your pen. Store a spare notebook in the drawer.|notebook
Explain how to arrange a picnic blanket before lunch.|Spread the blanket on level ground. Smooth the blanket before setting down plates.|blanket
Suggest a way to share a puzzle with a friend.|Choose a puzzle together. Sort the puzzle pieces by color.|puzzle
Describe how to keep a reusable bottle ready for outings.|Wash the bottle after each outing. Refill the bottle before leaving.|bottle
Offer a simple way to remember a borrowed item.|Write a note when you borrow it. Keep the note near your keys.|note
Explain how to organize a weekend photo sorting session.|Choose an album first. Fill the album slowly. Label the album afterward.|album
Suggest a routine for a small kitchen herb pot.|Place the pot near daylight. Turn the pot every few days.|pot
Describe two uses for an empty tray during a craft session.|A tray holds loose beads. The tray also catches paper scraps.|tray
Explain how to set up a chair for an afternoon of reading.|Move the chair near the window. Put a cushion on the chair.|chair
Suggest how to keep a ribbon ready for wrapping a gift.|Roll the ribbon loosely. Store the ribbon beside the wrapping paper.|ribbon
Describe a plan for tidying a shelf of board games.|Clear the shelf first. Dust the shelf. Return the games to the shelf.|shelf
Offer a way to prepare a towel for a day at the beach.|Fold the towel into your bag. Keep the towel away from damp sandals.|towel
Explain how to use a label when sorting storage containers.|Write a label for each category. Attach the label where it is easy to see.|label
Suggest a way to keep track of an ongoing knitting project.|Keep the yarn in a pouch. Put the yarn beside your pattern.|yarn
Describe a routine for choosing a walking route.|Pick a route through familiar streets. Adjust the route if you want a shorter walk.|route
Explain how to share a magazine around a household.|Leave the magazine on the side table. Pass the magazine along after reading.|magazine''',
    'word_count_interval': '''Give a tip for keeping reading glasses easy to locate.|Set your glasses in a small dish beside your usual reading chair.
Suggest an easy way to organize reusable shopping lists.|Keep a pad near the pantry and add items as they run out.
Explain how to make a breakfast tray feel welcoming.|Arrange a napkin beside the plate and leave room for a small flower.
Offer a way to make a closet shelf easier to use.|Group folded items by size so you can see what fits each space.
Recommend a simple activity for a short wait at home.|Sort a handful of postcards and choose one to display on your desk.
Describe a way to prepare a bag for a swimming trip.|Put your towel and goggles together before adding a separate pouch for dry clothes.
Suggest a tidy way to keep spare stationery.|Place spare pens and envelopes in a shallow box near your writing space.
Give a tip for arranging a shared craft table.|Leave an open space in the middle for supplies that everyone will use.
Explain how to remember a reusable mug before leaving home.|Set your mug beside your bag after washing it the evening before.
Offer a way to make a small patio more inviting.|Move a comfortable chair into the shade and add a pot of flowers.
Recommend two ways to organize spare gift bags.|Sort the bags by size.~Fold the smallest bags inside a larger one.
Suggest two preparations for a relaxed movie evening.|Place cushions where everyone can reach them.~Set out snacks before the movie begins.
Offer two ideas for displaying small travel keepsakes.|Group the keepsakes on a shallow shelf.~Leave space between items so each stands out.
Describe two ways to make laundry sorting easier.|Set out separate baskets for each group.~Keep an empty basket nearby for folded items.
Recommend two small tasks before a friend arrives for lunch.|Clear a place at the table.~Fill a jug with water for sharing.
Suggest two ways to tidy a bedside table.|Return finished reading material to the shelf.~Put small loose items in a dish.
Offer two preparations for an afternoon of watercolor painting.|Cover the table with a washable mat.~Set a water cup beside your brushes.
Give two ways to organize a collection of postcards.|Group the cards by the places shown.~Keep each group together with a paper band.
Suggest two preparations for a neighborhood plant swap.|Label the plants you plan to bring.~Pack the pots upright in a sturdy tray.
Describe two ways to make a reading basket useful.|Choose a few items you want to read.~Leave room for your glasses and a bookmark.''',
    'sentence_count': '''Suggest a small ritual for starting a journal.|Choose a comfortable seat and write down one thing you noticed today.
Explain how to share table space during a board game.|Keep snacks on a side table. Leave the middle clear for the game.
Offer a way to remember reusable cutlery for lunch.|Wash the cutlery after lunch. Dry it thoroughly. Return it to your bag.
Recommend a quiet activity for the end of a long day.|Arrange a few favorite photos in a small album.
Describe a way to keep a scarf from getting lost on outings.|Fold it into your bag when you take it off. Use the same pocket each time.
Suggest a simple plan for browsing a weekend market.|Bring a roomy bag. Walk through once before choosing anything. Leave time for a short break.
Give advice for setting up a jigsaw puzzle on a small table.|Use a flat board that you can move when the table is needed.
Explain how to make a shared notice board easier to read.|Remove notices that are no longer useful. Group the remaining notes by topic.
Offer a way to prepare for a casual sketching session.|Choose an object nearby. Gather a pencil and paper. Spend a moment noticing its shape.
Recommend a place for a collection of smooth stones.|Arrange the stones in a shallow dish where their colors are easy to see.
Give a friendly reminder to bring a tote on a shopping trip.|Put your tote beside your keys before you leave. Ready for the shops.
Offer a tip for keeping reusable napkins handy.|Fold the napkins after washing. Stack them near the plates. Set for supper.
Suggest an easy way to start sorting family photos.|Choose one small envelope of photos to begin. One batch at a time.
Describe how to set up a simple picnic spot.|Choose a level patch of shade. Spread out a mat. Lunch can wait a moment.
Offer a gentle reminder to return a borrowed baking tin.|Place the clean tin beside your bag. Back to its shelf soon.
Recommend a way to enjoy an evening on a balcony.|Bring out a comfortable cushion. Sit where you can watch the sky. A quiet pause.
Suggest a manageable first step for tidying a hobby drawer.|Take out the loose scraps and sort them into a small tray. Start with what fits.
Explain how to prepare a board game for the next gathering.|Gather the pieces into their bags. Fold the board carefully. Ready for another round.
Offer a pleasant way to end a weekend walk.|Choose a bench near the end of your route. Take a moment to look around. Home at an easy pace.
Recommend a way to keep a favorite recipe accessible.|Place a marker at the recipe before closing the book. Easy to find next time.''',
    'paragraph_count': '''Describe a simple arrangement for a morning tea break.|Set a cup beside a small plate.^Choose a comfortable place with room to rest your elbows.
Offer a plan for organizing a modest collection of board games.|Gather the loose pieces first.^Return each set to its own box.^Stack the boxes with their titles visible.
Explain how to prepare a basket for a park outing.|Put a cloth at the bottom of the basket.^Add the food containers and leave a space for napkins.
Suggest a way to make a tiny entryway easier to use.|Reserve one hook for the bag you use most.^Place a shallow tray nearby for small items.
Describe a gentle start to sorting old greeting cards.|Choose a small stack to look through.^Set aside the cards you want to display.^Keep the rest together in an envelope.
Offer a routine for putting away picnic supplies.|Shake out the blanket and let it air.^Wash the containers before returning everything to the basket.
Recommend a way to arrange a windowsill of small plants.|Put the tallest pots at the back.^Leave enough room to reach each pot when watering.
Explain how to set up a simple wrapping station.|Clear a space larger than the gift.^Lay out paper and tape.^Keep scissors within easy reach.
Suggest a plan for sharing a small bookcase.|Give each reader a shelf or a marked section.^Leave a shared space for books that everyone uses.
Describe a tidy way to store a partly finished craft project.|Gather the loose pieces into a pouch.^Place the pouch beside the main project in a shallow box.
Offer a routine for preparing a guest reading nook.|Set a soft cushion on the chair.^Place a few varied magazines on the nearby table.
Suggest a way to sort a collection of spare envelopes.|Group the envelopes by size.^Put the smallest group at the front.^Keep stamps in a separate little pocket.
Explain how to organize a shelf of reusable drink cups.|Pair each cup with its lid.^Arrange the pairs so they are easy to lift out.
Recommend a small plan for a weekend balcony cleanup.|Start by moving the pots aside.^Sweep the open space.^Return the pots with room to walk between them.
Describe how to prepare a box of supplies for making cards.|Gather plain card and colored paper.^Add glue and a few pens in a small pouch.
Suggest a way to arrange a collection of favorite recipes.|Group the recipes by the meals you use them for.^Place the most familiar recipes at the front.
Offer a routine for unpacking after a day trip.|Return unused snacks to the kitchen.^Set washable items aside.^Hang the empty bag where it can air.
Explain how to keep a shared puzzle corner organized.|Use small trays for sorted pieces.^Keep the puzzle board clear of unrelated items.
Recommend a way to prepare for a slow Sunday breakfast.|Set out the plates the evening before.^Choose a simple meal that leaves time to linger.
Describe a welcoming setup for a casual craft afternoon.|Put shared supplies in the center of the table.^Give each person a clear working space.^Keep a spare chair nearby.''',
    'bullet_count': '''Suggest items to keep in a small mending kit.|- needle~- thread~- spare buttons
Recommend simple activities for a short garden break.|- look for new leaves~- listen to nearby birds
Offer ways to sort a shelf of cookbooks.|- group by meal type~- arrange by height~- keep favorites together
Suggest items for a relaxed afternoon at the pond.|- folding mat~- water bottle
Recommend places to keep a spare pencil handy.|- beside the calendar~- in a shopping bag~- near the telephone
Offer ways to display a single favorite postcard.|- clip it to a small stand~- place it in a simple frame
Suggest tasks for refreshing a hobby workspace.|- clear paper scraps~- sort loose tools~- wipe the work surface
Recommend items for a simple outdoor sketching pouch.|- small sketchbook~- soft pencil~- eraser
Offer ways to keep a tablecloth neatly stored.|- fold it along the seams~- place it on a clear shelf
Suggest quiet activities for an early morning at home.|- read a short chapter~- sketch the view~- arrange a vase
Recommend things to check before lending a board game.|- count the pieces~- include the instructions
Offer small improvements for a shared coat area.|- add a low hook~- clear the floor~- label a spare basket
Suggest uses for leftover plain wrapping paper.|- cover a notebook~- make gift tags
Recommend items to pack for a knitting circle.|- current project~- spare yarn~- pattern notes
Offer ideas for arranging a small collection of shells.|- group similar shapes~- use a shallow dish~- leave space around each group
Suggest ways to prepare a table for a shared puzzle.|- remove loose papers~- set out sorting trays
Recommend small tasks for a rainy afternoon indoors.|- sort a drawer~- mend a loose button~- write a postcard
Offer ways to keep gardening gloves together.|- clip the pair together~- store them beside the watering can
Suggest items for a simple travel stationery pouch.|- folded note paper~- pencil~- small eraser
Recommend finishing touches for a picnic basket.|- tuck in napkins~- add a reusable cloth''',
    'json_only': '''Suggest a place to keep spare coasters.|A shallow drawer near the table.
Recommend an object to sketch on a quiet afternoon.|A pear on a plain plate.
Offer a name for a small neighborhood walking club.|The Wandering Acorns.
Suggest a snack to share during a puzzle session.|Sliced apples in a small bowl.
Recommend a simple way to display dried flowers.|Stand them in a narrow vase.
Offer a color pairing for a handmade bookmark.|Soft green with warm cream.
Suggest a useful item for a picnic bag.|A washable cloth for the table.
Recommend a spot for a household message pad.|Beside the place where you keep your keys.
Offer a theme for a casual photo walk.|Interesting doorways and window boxes.
Suggest a small gift for someone who likes journaling.|A sheet of decorative paper tabs.
Recommend a container for loose craft beads.|A clear jar with a secure lid.
Offer a simple idea for labeling homemade jam.|A paper tag tied around the lid.
Suggest a pleasant place to pause on a neighborhood walk.|A bench beside a small garden.
Recommend an item to keep beside a crossword book.|A soft pencil with a clean eraser.
Offer a way to mark a seedling tray.|Use wooden labels with clear plant names.
Suggest a name for a shared household recipe folder.|The Everyday Table.
Recommend a small addition to an outdoor reading seat.|A light blanket for the evening.
Offer a way to display a handmade card.|Rest it on a narrow picture ledge.
Suggest a simple subject for a first watercolor sketch.|A single leaf on white paper.
Recommend a useful extra for a reusable shopping bag.|A small pouch for loose produce tags.''',
    'lowercase_only': '''Give a friendly message for a shared breakfast table.|help yourself to a plate and find a comfortable seat.
Suggest a small habit for keeping a sketchbook nearby.|leave your sketchbook beside the chair where you often sit.
Recommend a way to store rolls of decorative tape.|stand the rolls together in a shallow tray.
Offer a pleasant invitation to a weekend walk.|join us for a gentle walk and a little fresh air.
Explain how to make a small craft shelf easier to browse.|put clear labels on the boxes and face them outward.
Suggest a simple use for a spare cotton cloth.|line a basket with it before adding fresh bread.
Give a reminder to put a borrowed puzzle back together.|gather every piece and return the lid before putting the box away.
Recommend a quiet activity while bread dough rests.|write a short note to someone you have been meaning to thank.
Offer a way to arrange a bedside stack of books.|keep the current book on top and return finished ones to the shelf.
Suggest a welcoming detail for a shared garden table.|put a small jar of flowers in the middle.
Recommend a color for a plain picnic napkin.|soft blue
Suggest an item to include in a paper craft box.|a small glue stick
Offer a name for a casual neighborhood book exchange.|the turning page
Recommend a topping for a bowl of morning oats.|sliced pear
Suggest a subject for a rainy window sketch.|droplets on the glass
Offer a small task for the start of a tidying session.|clear one corner of the desk
Recommend an item for a comfortable park visit.|a folded sitting mat
Suggest a gentle activity after a long afternoon of errands.|listen to a familiar album
Offer a place to store a reusable bread bag.|beside the bread basket
Recommend a simple decoration for a handmade envelope.|a small leaf drawing''',
    'required_suffix': '''Write a warm invitation to a casual tea break.|Come sit with us for a cup of tea. There is room at the table.|There is room at the table.
Offer encouragement to someone sorting a crowded drawer.|Choose a small section to begin with. A little space is a start.|A little space is a start.
Give a friendly reminder about a weekend park meetup.|Bring a mat if you would like to sit on the grass. See you by the pond.|See you by the pond.
Suggest an easy way to begin a photo album.|Pick a few photos from one afternoon. Leave a page for later.|Leave a page for later.
Write a short note to accompany a borrowed baking dish.|Thank you for lending the dish for our lunch. Back with thanks.|Back with thanks.
Offer a gentle invitation to a slow neighborhood stroll.|Come along whenever you are ready. We can take our time.|We can take our time.
Give encouragement for starting a small sewing project.|Choose a simple shape and gather your supplies. One stitch to begin.|One stitch to begin.
Suggest a closing note for a household shopping list.|Check the pantry before heading out. The tote is by the door.|The tote is by the door.
Write a friendly message for a shared craft afternoon.|Bring whatever you are working on and pull up a chair. Space for every project.|Space for every project.
Offer a reminder to take a break while organizing a shelf.|Pause after the first section and stretch your hands. The rest can wait.|The rest can wait.
Write an encouraging note for someone learning to sketch.|Keep your pencil moving while you notice the shape. A pencil can wander.|A pencil can wander.
Suggest a friendly message for a basket of shared fruit.|Take a piece of fruit for your afternoon break. Fruit for the taking.|Fruit for the taking.
Offer a short note for a packet of flower seeds being shared.|These seeds can brighten a sunny corner. A few seeds to share.|A few seeds to share.
Give a warm invitation to a board game evening.|Choose a game from the shelf when you arrive. A game for everyone.|A game for everyone.
Write a note for a borrowed umbrella being returned.|Your umbrella kept the drizzle away on my walk. One umbrella returned.|One umbrella returned.
Offer encouragement for building a small daily reading habit.|Keep a book beside your favorite chair. A book is good company.|A book is good company.
Suggest a friendly label for a box of spare yarn.|Take some yarn for your next little project. Yarn looking for a home.|Yarn looking for a home.
Write a cheerful message for a jar of homemade cookies.|Enjoy a cookie with your afternoon drink. A cookie for a pause.|A cookie for a pause.
Offer a note for a shared collection of spare plant pots.|Choose a pot that suits your next seedling. A pot for new leaves.|A pot for new leaves.
Give encouragement to someone assembling a scrapbook.|Choose a photo that brings back a pleasant moment. One photo starts the story.|One photo starts the story.''',
}

# Ten authored pairs per family; literal arguments are substituted after selection.
# The words describing checker semantics are part of each clause, not task content.
CLAUSES = {
'required_word': [
('Work the whole word "{word}" into the answer at least once, with any capitalization.', 'Somewhere in the answer, use "{word}" as a complete word; letter case does not matter.'),
('Make room for at least one occurrence of the word "{word}", ignoring case.', 'The word "{word}" needs to appear in full at least once; uppercase and lowercase both count.'),
('Ensure the answer contains "{word}" as a standalone word, in any letter case.', 'Use at least one complete instance of "{word}"; its capitalization is unrestricted.'),
('Include a complete word matching "{word}" without regard to capitalization.', 'At least one whole word in the reply must be "{word}" when case is ignored.'),
('Use the word "{word}" somewhere in the reply; matching is case insensitive and whole-word only.', 'The reply needs a whole-word occurrence of "{word}"; changing letter case is fine.'),
('Find a place for the complete word "{word}" in your reply, regardless of case.', 'Let "{word}" occur at least once as a full word, using whichever capitalization you prefer.'),
('Your answer needs at least one full instance of "{word}" in any capitalization.', 'Put the whole word "{word}" in the answer one or more times; case is not checked.'),
('Mention "{word}" as a whole word at least once; capitalizing it is allowed.', 'A complete occurrence of "{word}" must be present, with uppercase and lowercase treated alike.'),
('Have "{word}" appear as a complete word in the response; case makes no difference.', 'Give the response at least one whole-word use of "{word}", disregarding letter case.'),
('The complete word "{word}" must occur at least once, irrespective of capitalization.', 'Use "{word}" one or more times as an entire word; any casing counts.'),
],
'forbidden_word': [
('Leave out the whole word "{word}" in every capitalization.', 'No complete word in the reply may match "{word}" when case is ignored.'),
('Keep the answer free of the standalone word "{word}", regardless of letter case.', 'Exclude "{word}" as a whole word; uppercase variants are excluded too.'),
('Avoid every whole-word occurrence of "{word}", with case ignored.', 'The full word "{word}" must be absent in both uppercase and lowercase forms.'),
('Do not use "{word}" as a complete word, however it is capitalized.', 'Omit the whole word "{word}"; changes to its letter case do not make it acceptable.'),
('No whole word may equal "{word}" under a case-insensitive comparison.', 'Keep all complete instances of "{word}" out of the response, independent of capitalization.'),
('Leave the word "{word}" unused; this applies to whole words in any case.', 'Avoid "{word}" whenever it would be an entire word, including differently capitalized forms.'),
('Exclude the complete word "{word}" from the reply, irrespective of case.', 'There must be zero whole-word matches for "{word}" after ignoring capitalization.'),
('Keep "{word}" out as a standalone word, whether capitalized or not.', 'Do not write the full word "{word}" in any combination of uppercase and lowercase.'),
('All case variants of the whole word "{word}" are off limits.', 'The answer must omit "{word}" as a complete word, with no exception for capitalization.'),
('Use no complete instance of "{word}"; case does not affect this restriction.', 'A case-insensitive search for the whole word "{word}" must find nothing in the answer.'),
],
'exact_word_frequency': [
('Use the whole word "{word}" exactly {count} times, ignoring capitalization.', 'There must be {count} complete occurrences of "{word}" in total, with all letter cases counted alike.'),
('Limit the full word "{word}" to exactly {count} appearances across the reply; case is ignored.', 'Count "{word}" as a whole word without regard to case and make that count exactly {count}.'),
('Make exactly {count} whole-word uses of "{word}" in any capitalization.', 'Across the answer, "{word}" must occur as a complete word {count} times; case does not matter.'),
('The total for the standalone word "{word}" must be {count}, counted case insensitively.', 'Use "{word}" as an entire word on exactly {count} occasions, regardless of capitalization.'),
('Write the complete word "{word}" {count} times overall, treating upper and lower case alike.', 'Your reply needs exactly {count} case-insensitive whole-word matches for "{word}".'),
('Give "{word}" precisely {count} full-word appearances; letter case is unrestricted.', 'Exactly {count} whole words must match "{word}" after capitalization is ignored.'),
('Keep the case-insensitive count of the whole word "{word}" at exactly {count}.', 'Let "{word}" appear {count} times as a complete word; count every capitalization together.'),
('Arrange the wording so that "{word}" occurs exactly {count} times as a whole word, in any case.', 'The reply must contain precisely {count} standalone instances of "{word}", irrespective of letter case.'),
('Use exactly {count} copies of the word "{word}" as complete words; capitalization does not change the count.', 'There should be {count} whole-word occurrences of "{word}" altogether, counted without case distinctions.'),
('Make the whole-word tally for "{word}" equal {count}, with every case variant included.', 'Write "{word}" as a full word exactly {count} times in total; upper and lower case both qualify.'),
],
'word_count_interval': [
('Keep the reply between {min} and {max} words inclusive, counting whitespace-separated units.', 'Count units separated by whitespace as words and use at least {min} but no more than {max}.'),
('Use {min} to {max} whitespace-delimited words, including both limits.', 'The reply needs a word total in the inclusive range {min} through {max}; split on whitespace to count.'),
('Set the length at no fewer than {min} and no more than {max} words; whitespace separates words.', 'Counting each whitespace-separated unit as one word, stay within {min}–{max}, endpoints included.'),
('Aim for the inclusive word range {min}–{max}, using whitespace boundaries for the count.', 'Keep the number of whitespace-separated words at least {min} and at most {max}.'),
('The whole answer must have {min} through {max} words inclusive; use a whitespace split.', 'Divide the answer at whitespace and ensure that the resulting word count lies from {min} to {max}, including either endpoint.'),
('Write within a {min}–{max} word budget, inclusive; a word is one whitespace-separated unit.', 'Use at least {min} words and at most {max}, where whitespace boundaries determine the units.'),
('Count words by whitespace and keep their total in the closed interval {min} to {max}.', 'The response length must be between {min} and {max} whitespace-delimited words, with both bounds permitted.'),
('Allow {min}–{max} words for the entire reply, inclusive, counting via whitespace separation.', 'Using whitespace to separate words, make the total no smaller than {min} and no larger than {max}.'),
('Use an inclusive range of {min} to {max} words; count every whitespace-delimited unit.', 'A whitespace-based word tally must fall between {min} and {max}, and either boundary is acceptable.'),
('Fit the answer into {min} through {max} words, endpoints included; whitespace defines the word divisions.', 'Ensure a total of at least {min} and at most {max} words when splitting the answer on whitespace.'),
],
'sentence_count': [
('Use exactly {count} sentences; split sentences at whitespace following a period, question mark or exclamation mark.', 'Make {count} sentence segments in total, taking whitespace after . ? or ! as the boundary.'),
('Keep the sentence total at {count}, with a boundary at whitespace after . ? or !.', 'There must be exactly {count} sentences when text is split wherever whitespace follows a period, question mark or exclamation mark.'),
('Write {count} sentences overall; a sentence break is whitespace immediately after . ? or !.', 'Divide the reply at whitespace after a period, question mark or exclamation mark, yielding exactly {count} sentences.'),
('Give the answer {count} sentences, counting breaks at whitespace following . ? or !.', 'The reply must split into {count} sentence segments at whitespace that comes after a period, question mark or exclamation mark.'),
('The entire response needs exactly {count} sentences; whitespace after . ? or ! separates them.', 'Count sentence boundaries at whitespace following a period, question mark or exclamation mark and make the total {count}.'),
('Use a total of {count} sentences under the rule that whitespace after . ? or ! starts a new sentence.', 'There should be {count} sentence segments, separated where whitespace follows a period, question mark or exclamation mark.'),
('Make the sentence count exactly {count}; recognize boundaries only at whitespace after . ? or !.', 'Splitting on whitespace after a period, question mark or exclamation mark must give {count} sentences.'),
('Limit the response to exactly {count} sentences, defined by whitespace after . ? or !.', 'Arrange the reply as {count} sentence segments; their dividing points are whitespace following a period, question mark or exclamation mark.'),
('Have {count} sentences in the reply; count a split wherever whitespace follows . ? or !.', 'With sentence breaks placed at whitespace after a period, question mark or exclamation mark, the total must be {count}.'),
('Set the number of sentences to {count}, splitting at whitespace immediately following . ? or !.', 'Produce exactly {count} segments when the reply is divided at whitespace after a period, question mark or exclamation mark.'),
],
'paragraph_count': [
('Arrange the answer in exactly {count} nonempty paragraphs separated by blank lines.', 'Use blank lines between paragraphs and make exactly {count} nonempty blocks.'),
('Create {count} paragraphs in all, each nonempty, with blank lines marking the breaks.', 'The reply must have exactly {count} nonempty text blocks when split at blank lines.'),
('Use blank-line paragraph breaks and give the response exactly {count} nonempty paragraphs.', 'Divide the answer into {count} nonempty paragraphs by leaving a blank line between them.'),
('Write exactly {count} filled paragraphs, separating them with blank lines.', 'Make the blank-line-separated paragraph total {count}; no paragraph may be empty.'),
('Keep the answer to {count} nonempty paragraphs, using a blank line between neighboring paragraphs.', 'There must be precisely {count} nonempty paragraphs with blank lines as their separators.'),
('Set out {count} nonempty paragraphs; leave blank lines at the paragraph boundaries.', 'Blank lines must divide the response into exactly {count} nonempty paragraph blocks.'),
('The reply needs {count} nonempty paragraphs in total, separated with blank lines.', 'Place blank lines between {count} paragraphs and put text in every paragraph.'),
('Give the answer a total of {count} paragraphs; each must contain text and be separated by a blank line.', 'Use exactly {count} text-containing paragraphs with blank lines between them.'),
('Break the response into exactly {count} nonempty paragraphs at blank lines.', 'The number of nonempty paragraphs must equal {count}, counting blank lines as the breaks.'),
('Make {count} nonempty paragraph blocks, leaving a blank line between blocks.', 'Format the reply as exactly {count} nonempty paragraphs divided by blank lines.'),
],
'bullet_count': [
('Include exactly {count} bullet lines, each starting with - or * followed by whitespace and text.', 'Count lines beginning with a hyphen or asterisk plus whitespace and text as bullets; there must be {count}.'),
('Use {count} bullet lines in total; begin each with - or * and then whitespace before its text.', 'The total of bullet lines must be {count}, using a hyphen or asterisk followed by whitespace and nonempty text.'),
('Provide exactly {count} bullets, recognized as lines starting with - or * plus whitespace and text.', 'Have {count} lines whose bullet marker is a hyphen or asterisk separated from its text by whitespace.'),
('Make the bullet count {count}; a bullet line starts with - or *, whitespace, then text.', 'Exactly {count} lines must carry a leading hyphen or asterisk with whitespace before the content.'),
('There must be {count} bullet lines, marked by an initial - or * followed by whitespace and text.', 'Use exactly {count} lines that start with a hyphen or asterisk, then whitespace, then nonempty content.'),
('Give the reply {count} bullets with - or * at the beginning of each line and whitespace before the text.', 'Exactly {count} bullet lines are needed; mark them with a hyphen or asterisk followed by whitespace and content.'),
('Set the number of bullet lines to {count}, using - or * followed by whitespace and text.', 'Count {count} lines beginning with a hyphen or asterisk and a whitespace gap before their content.'),
('Add exactly {count} lines of bullets; each begins with - or *, then whitespace, then text.', 'Use a leading hyphen or asterisk and whitespace to mark each nonempty bullet line, making {count} such lines.'),
('Keep precisely {count} bullet lines, where - or * precedes whitespace and text at the line start.', 'The reply must contain {count} bullet lines identified by an initial hyphen or asterisk, whitespace and content.'),
('Write {count} bullet lines overall, starting each with - or * and whitespace before its content.', 'There should be exactly {count} lines marked as bullets by a leading hyphen or asterisk followed by whitespace and text.'),
],
'json_only': [
('Let the entire reply be one valid JSON value, with no surrounding prose or code fence.', 'Return a single valid JSON value as the complete answer; omit prose outside it and any code fencing.'),
('Write only one valid JSON value; do not wrap it in a code block or explanatory text.', 'The full response must parse as a single JSON value without surrounding explanation or a fenced block.'),
('Use a single valid JSON value for the whole answer, without an introduction, ending note or code fence.', 'Everything you return must belong to one valid JSON value; add neither framing prose nor code fences.'),
('The answer must consist solely of one valid JSON value, unaccompanied by prose or code fencing.', 'Provide one valid JSON value and nothing around it, including no code fence or commentary.'),
('Encode the full reply as a lone valid JSON value; exclude external prose and fenced formatting.', 'A single valid JSON value must occupy the entire response, with no added explanation or code-block wrapper.'),
('Respond with one valid JSON value only, leaving out code fences and any text before or after it.', 'Use the response solely for a single valid JSON value, without fencing or surrounding prose.'),
('Make the complete response valid JSON containing one value, with no prose wrapper or code fence.', 'The entire answer should be just one valid JSON value; no surrounding commentary or fenced block is allowed.'),
('Restrict the reply to a single valid JSON value, with neither code fencing nor additional prose.', 'Supply one valid JSON value as all of your output, free of external explanation and code fences.'),
('Present one valid JSON value as the whole reply; do not add prose around it or place it in a code fence.', 'Only a single valid JSON value may be returned, without a surrounding narrative or fenced formatting.'),
('Give the answer entirely as one valid JSON value, omitting any outside prose and code fences.', 'Return nothing beyond a single valid JSON value; leave off commentary and code-block fencing.'),
],
'lowercase_only': [
('Use lowercase for every letter in the response.', 'Every letter you write must be in lowercase.'),
('Keep all letters lowercase throughout the answer.', 'Write the whole answer without any uppercase letters.'),
('Make the reply entirely lowercase wherever letters occur.', 'All alphabetic characters in the reply must be lowercase.'),
('No uppercase letters may appear anywhere in the response.', 'Keep every letter of the response in its lowercase form.'),
('Use only lowercase letter forms across the entire reply.', 'Do not capitalize any letters in any part of the reply.'),
('Leave every letter uncapitalized, including sentence beginnings.', 'Write all letters in lowercase.'),
('The response must contain no capital letters.', 'Write every letter in lowercase across the whole response.'),
('Keep letter case uniformly lowercase from beginning to end.', 'All letters throughout the answer must be in lowercase.'),
('Write any letters in lowercase; capitals are not permitted.', 'Use lowercase letters exclusively wherever the reply contains letters.'),
('Avoid capitalization anywhere in the answer.', 'Ensure that no letter in the answer is uppercase.'),
],
'required_suffix': [
('Make the exact text "{suffix}" the very end of the reply, with nothing after it, including whitespace.', 'The last characters must be "{suffix}" exactly; append no text or whitespace afterward.'),
('Close with the literal suffix "{suffix}" and leave no trailing characters, even spaces or line breaks.', 'End the response at "{suffix}" with exact spelling and case; nothing, including whitespace, may follow.'),
('Finish at the exact characters "{suffix}"; do not add anything after them, even whitespace.', 'The reply must terminate with "{suffix}" verbatim, with no trailing text, spaces or newlines.'),
('Use "{suffix}" as the exact ending, with no characters beyond it, including whitespace.', 'Stop immediately after the literal text "{suffix}"; no subsequent spaces, newlines or other characters are allowed.'),
('The response must end with "{suffix}" exactly and have no following text or whitespace.', 'Place the literal suffix "{suffix}" at the end, preserving its case and leaving nothing after it, even whitespace.'),
('Put the exact suffix "{suffix}" last; do not follow it with a space, line break or any other character.', 'Terminate the answer immediately after "{suffix}" as written, without trailing whitespace or text.'),
('Stop the reply at the literal ending "{suffix}", with no trailing characters of any kind.', 'The exact text "{suffix}" must be the final characters; no whitespace or other text may come after it.'),
('End on "{suffix}" verbatim, adding no text, spaces or line breaks afterward.', 'Make "{suffix}" the precise suffix of the response and leave no following characters, including whitespace.'),
('Let "{suffix}" be the final literal text; nothing at all, even whitespace, may follow.', 'Close the reply with exactly "{suffix}" and omit all trailing text and whitespace.'),
('Use the exact final characters "{suffix}" and stop there, without trailing whitespace or text.', 'The response must finish at the verbatim suffix "{suffix}", with no spaces, newlines or other characters afterward.'),
],
}


def tokens(text):
    return re.findall(r'\b\w+\b', text.casefold())


def render_v1():
    rows = []
    for family, block in DATA.items():
        entries = block.splitlines()
        assert len(entries) == 20, (family, len(entries))
        for index, entry in enumerate(entries):
            fields = entry.split('|')
            content, witness = fields[:2]
            witness = witness.replace('~', '\n').replace('^', '\n\n')
            arg = fields[2] if len(fields) == 3 else None
            if family in ('required_word', 'forbidden_word'):
                params = {'word': arg}
            elif family == 'exact_word_frequency':
                params = {'word': arg, 'count': tokens(witness).count(arg)}
                if index >= 10:
                    witness = witness.replace('. ', '.\n\n', 1)
            elif family == 'word_count_interval':
                if index >= 10:
                    witness = '\n'.join('- ' + line for line in witness.splitlines())
                n = len(witness.split())
                params = {'min': n - 3, 'max': n + 4}
            elif family == 'sentence_count':
                params = {'count': len(re.split(r'(?<=[.!?])\s+', witness.strip()))}
            elif family == 'paragraph_count':
                params = {'count': len(witness.split('\n\n'))}
            elif family == 'bullet_count':
                params = {'count': len(witness.splitlines())}
            elif family == 'required_suffix':
                params = {'suffix': arg}
            else:
                params = {}
            if family == 'json_only':
                witness = json.dumps({'suggestion': witness}, ensure_ascii=False)
            if family == 'lowercase_only' and index >= 10:
                witness = json.dumps(witness, ensure_ascii=False)
            constraints = [{'family': family, 'parameters': params}]
            if index >= 10:
                secondary = {
                    'required_word': ('word_count_interval', {'min': 6, 'max': 20}),
                    'forbidden_word': ('sentence_count', {'count': len(re.split(r'(?<=[.!?])\s+', witness.strip()))}),
                    'exact_word_frequency': ('paragraph_count', {'count': 2}),
                    'word_count_interval': ('bullet_count', {'count': 2}),
                    'sentence_count': ('required_suffix', {'suffix': re.split(r'(?<=[.!?])\s+', witness)[-1]}),
                    'paragraph_count': ('forbidden_word', {'word': ['perfect', 'amazing', 'urgent', 'luxury', 'instant', 'effortless', 'premium', 'ultimate', 'spectacular', 'mandatory'][index - 10]}),
                    'bullet_count': ('lowercase_only', {}),
                    'json_only': ('required_word', {'word': next(w for w in tokens(fields[1]) if len(w) >= 4)}),
                    'lowercase_only': ('json_only', {}),
                    'required_suffix': ('exact_word_frequency', {'word': ['pencil', 'fruit', 'seeds', 'game', 'umbrella', 'book', 'yarn', 'cookie', 'pot', 'photo'][index - 10], 'count': 2}),
                }[family]
                constraints.append({'family': secondary[0], 'parameters': secondary[1]})
            canonical, paraphrase = [], []
            for offset, constraint in enumerate(constraints):
                pair = CLAUSES[constraint['family']][(index + offset * 3) % 10]
                canonical.append(pair[0].format(**constraint['parameters']))
                paraphrase.append(pair[1].format(**constraint['parameters']))
            rows.append({
                'id': f'sp-{family}-{index + 1:03d}',
                'content': content,
                'constraints': constraints,
                'canonical_instruction': ' '.join(canonical),
                'paraphrase_instruction': ' '.join(paraphrase),
                'witness': witness,
            })
    return rows


# Round 3: paraphrases carry only the information in the canonical instruction.
# Operational details remain in the checker contract, not either prompt arm.
FORBIDDEN_REPLACEMENTS = {
    'front': 'podium', 'craft': 'origami', 'clean': 'spotless', 'tea': 'cappuccino',
    'book': 'encyclopedia', 'game': 'charades', 'mess': 'squalor', 'box': 'shoebox',
    'ball': 'frisbee', 'perfect': 'flawless', 'urgent': 'hasty',
    'instant': 'immediate', 'ultimate': 'superb',
}
PARAPHRASE_OVERRIDES = {
'required_word': [
'Work the word "{word}" into your answer.',
'Make sure your reply mentions "{word}".',
'Use "{word}" somewhere in what you write.',
'Your answer needs to include the keyword "{word}".',
'Find a place for "{word}" in the reply.',
'Bring the word "{word}" into your response.',
'Let "{word}" appear somewhere in the answer.',
'Mention "{word}" when responding.',
'Put the keyword "{word}" into the text of your reply.',
'Incorporate "{word}" into what you say.',
],
'forbidden_word': [
'Leave the word "{word}" out of your answer.',
'Avoid using "{word}" in the reply.',
'Keep "{word}" out of what you write.',
'The response must omit the keyword "{word}".',
'Refrain from mentioning "{word}".',
'Choose wording that does not use "{word}".',
'Exclude the word "{word}" from your answer.',
'Write your reply without "{word}".',
'Do not put "{word}" anywhere in the answer.',
'Leave "{word}" unmentioned in the response.',
],
'word_frequency': [
'The word "{word}" must occur exactly {count} times in the answer.',
'Use the word "{word}" on exactly {count} occasions in your reply.',
'Let the word "{word}" appear {count} times, no more and no fewer.',
'Give the word "{word}" precisely {count} appearances in the answer.',
'Write the word "{word}" exactly {count} times altogether.',
'Keep the number of uses of the word "{word}" at {count}.',
'There must be exactly {count} uses of the word "{word}" in your response.',
'Make the total number of occurrences of the word "{word}" equal {count}.',
'Use the word "{word}" {count} times overall, and no additional times.',
'Have the word "{word}" appear precisely {count} times across the reply.',
],
'word_count_interval': [
'Use no fewer than {min} words and no more than {max}.',
'Keep your reply between {min} and {max} words, including either limit.',
'The answer needs at least {min} words, with {max} as the maximum.',
'Write a reply of {min} to {max} words, inclusive.',
'Give the response a length of at least {min} and at most {max} words.',
'Your word total must fall from {min} through {max}, inclusive.',
'Fit the answer within an inclusive word range of {min} to {max}.',
'Use {min} words or more, but stop at {max} words or fewer.',
'Make the reply no shorter than {min} words and no longer than {max}.',
'Allow at least {min} words for the response, without exceeding {max}.',
],
'sentence_count': [
'Write {count} sentences in total, with neither an extra sentence nor a missing one.',
'The reply needs precisely {count} sentences overall.',
'Let the answer contain {count} sentences, no more and no fewer.',
'Make the complete response consist of {count} sentences.',
'The total number of sentences must equal {count}.',
'Give the reply a sentence total of {count}.',
'Arrange what you say into {count} sentences altogether.',
'Keep the number of sentences at precisely {count}.',
'Produce a response whose sentence tally is {count}.',
'There should be exactly {count} sentences across the whole answer.',
],
'paragraph_count': [
'Write exactly {count} paragraphs with "* * *" between them.',
'Arrange the answer as {count} paragraphs, placing "* * *" between neighboring paragraphs.',
'Divide your response into exactly {count} paragraphs separated by "* * *".',
'Give the reply {count} paragraphs in all, using "* * *" to separate them.',
'Produce exactly {count} paragraphs and mark their boundaries with "* * *".',
'Make the answer consist of {count} paragraphs with the divider "* * *" between them.',
'There must be {count} paragraphs altogether, separated by "* * *".',
'Use exactly {count} paragraphs, each separated from the next by "* * *".',
'Produce exactly {count} paragraphs, putting "* * *" between them.',
'Present the reply in {count} paragraphs and separate those paragraphs with "* * *".',
],
'bullet_count': [
'Include exactly {count} Markdown bullet points in your answer.',
'Give precisely {count} Markdown bullet points in your response.',
'Use {count} Markdown bullet points, no more and no fewer.',
'Write a reply containing exactly {count} Markdown bullet points.',
'Produce exactly {count} Markdown bullet points for your answer.',
'The response needs a total of {count} Markdown bullet points.',
'Make the number of Markdown bullet points in the reply equal {count}.',
'Provide exactly {count} Markdown bullet points in your reply.',
'Present precisely {count} Markdown bullet points in the answer.',
'Give your reply {count} Markdown bullet points in all.',
],
'json_only': [
'Present your entire answer in JSON.',
'Make JSON the format of your whole reply.',
'Use JSON for the complete response.',
'Give the full answer as JSON.',
'Express everything in your response as JSON.',
'Render the whole reply in JSON format.',
'Let the complete answer take the form of JSON.',
'Use the JSON format throughout your reply.',
'Provide your entire response as JSON.',
'Write the complete reply using JSON.',
],
'required_suffix': [
'Let "{suffix}" be the exact ending of your answer.',
'Use "{suffix}" verbatim as the ending of your reply.',
'The reply must end with precisely "{suffix}".',
'Close the answer with the exact phrase "{suffix}".',
'Make "{suffix}" the final phrase of the response, exactly as written.',
'Put the phrase "{suffix}" at the end of your reply, unchanged.',
'Use exactly "{suffix}" to conclude the answer.',
'End your reply on the phrase "{suffix}", written exactly that way.',
'The response must close with "{suffix}" verbatim.',
'Give the answer the exact ending "{suffix}".',
],
}


def render():
    import sys
    sys.path.insert(0, str(ROOT.parent))
    import templates
    rows = []
    for old in render_v1():
        index = int(old['id'].rsplit('-', 1)[1]) - 1
        constraints = [dict(family=('word_frequency' if c['family'] == 'exact_word_frequency' else c['family']), args=c['parameters']) for c in old['constraints']]
        for c in constraints:
            if c['family'] == 'forbidden_word':
                c['args']['word'] = FORBIDDEN_REPLACEMENTS.get(c['args']['word'], c['args']['word'])
        witness = old['witness']
        if any(c['family'] == 'paragraph_count' for c in constraints):
            witness = witness.replace('\n\n', '\n* * *\n')
        paraphrases = []
        for offset, c in enumerate(constraints):
            family, args = c['family'], c['args']
            position = (index + offset * 3) % 10
            if family in PARAPHRASE_OVERRIDES:
                clause = PARAPHRASE_OVERRIDES[family][position].format(**args)
            else:
                key = 'exact_word_frequency' if family == 'word_frequency' else family
                clause = CLAUSES[key][position][1].format(**args)
            if family == 'lowercase_only': clause += ' Keep the response in English.'
            paraphrases.append(clause)
        canonical, bindings = templates.render(constraints)
        paraphrase = ' '.join(paraphrases)
        row = dict(schema='sp-source-v2', request_id=old['id'], primary_family=constraints[0]['family'],
                   content=old['content'], constraints=constraints,
                   canonical=dict(clause=canonical, clauses=[b['clause'] for b in bindings], template_bindings=bindings,
                                  prompt=old['content'] + '\n\n' + canonical),
                   paraphrase=dict(clause=paraphrase, clauses=paraphrases,
                                   prompt=old['content'] + '\n\n' + paraphrase), witness=witness,
                   mutations=[], reviews=[])
        rows.append(row)
    return rows


def source_bytes():
    return ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in render()).encode('utf-8')


if __name__ == '__main__':
    (ROOT / 'requests.jsonl').write_bytes(source_bytes())
    print('Wrote 200 template/paraphrase pairs (400 prompts), source schema sp-source-v2.')
