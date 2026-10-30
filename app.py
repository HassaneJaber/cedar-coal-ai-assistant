import logging
import chainlit as cl
from assistant import Assistant, Provider
from storage import Storage

logging.basicConfig(level=logging.INFO)
for logger in ("httpx", "httpx2", "openai"):
    logging.getLogger(logger).setLevel(logging.WARNING)
provider = None
storage = None
setup_error = None
try:
    storage = Storage()
    provider = Provider()
except (RuntimeError, ValueError) as exc:
    setup_error = str(exc)

@cl.set_starters
async def starters():
    return [cl.Starter(label=label, message=message) for label, message in [
        ("See the menu", "Show me your menu highlights."),
        ("Book a table", "I'd like to book a table."),
        ("Delivery areas", "Where do you deliver?"),
        ("شو في نباتي؟", "شو في نباتي؟")]]

@cl.on_chat_start
async def start():
    if provider:
        cl.user_session.set("assistant", Assistant(provider, storage))
    await cl.Message(content=(
        "Welcome to Cedar & Coal 🌿 — a fictional restaurant demo. I can help with the menu, hours, delivery and bookings.\n\n"
        "أهلا وسهلا! كيف فيني ساعدك؟ فيك تسأل عن المنيو، الدوام، التوصيل، أو تحجز طاولة. هيدا مطعم تجريبي.\n\n"
        "Bienvenue ! Menu, horaires, livraison et réservations — un restaurant de démonstration."
    ), actions=[cl.Action(name="quick_start", payload={"message": item.message}, label=item.label)
                for item in await starters()]).send()
    if setup_error:
        await cl.Message(content=setup_error).send()

@cl.on_message
async def message(message: cl.Message):
    bot = cl.user_session.get("assistant")
    if not bot:
        await cl.Message(content=setup_error or "Please restart the app.").send()
        return
    reply = await bot.reply(message.content)
    actions = []
    if bot.pending:
        # Snapshot prevents old buttons from confirming a different booking.
        payload = {"booking": dict(bot.pending)}
        actions = [cl.Action(name="confirm_booking", payload=payload, label=bot.button_labels()[0]),
                   cl.Action(name="cancel_booking", payload=payload, label=bot.button_labels()[1])]
    await cl.Message(content=reply, actions=actions).send()

@cl.action_callback("confirm_booking")
async def confirm(action):
    bot = cl.user_session.get("assistant")
    if bot and bot.pending == action.payload.get("booking"):
        await cl.Message(content=await bot.confirm()).send()
    else:
        await cl.Message(content="This confirmation has expired. Please review the latest details.").send()
    await action.remove()

@cl.action_callback("cancel_booking")
async def cancel(action):
    bot = cl.user_session.get("assistant")
    if bot and bot.pending == action.payload.get("booking"):
        await cl.Message(content=await bot.cancel()).send()
    await action.remove()

@cl.action_callback("quick_start")
async def quick_start(action):
    request = cl.Message(content=action.payload["message"], author="You", type="user_message")
    await request.send()
    await message(request)
