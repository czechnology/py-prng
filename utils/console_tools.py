def query_yes_no_abort(question, default=True):
    """Ask a yes/no/abort question via input() and return the answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be True (yes, the default), False (no) or None (meaning
        an answer is required of the user).

    The "answer" return value is one of "yes" or "no".
    """
    # Based on recipe from http://code.activestate.com/recipes/577058/

    if default is None:
        prompt = " ([y]es/[n]o/[a]bort) "
    elif default == True:
        prompt = " ([Y]es/[n]o/[a]bort) "
    elif default == False:
        prompt = " ([y]es/[N]o/[a]bort) "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        choice = input(question + prompt).lower()
        if default is not None and choice == '':
            return default
        elif choice in ('y', 'yes'):
            return True
        elif choice in ('n', 'no'):
            return False
        elif choice in ('a', 'abort'):
            return None
        else:
            print("Please respond with y/yes n/no")
