import constants


def itemid_source_query(ids):
    from_chart = [x for x in ids if x in constants.TOP_CHARTEVENTS_ITEMID]
    from_lab = [x for x in ids if x in constants.TOP_LABEVENTS_ITEMID]
    return from_chart, from_lab


def remove_elements(main_list, remove_list):
    new_list = [elem for elem in main_list if elem not in remove_list]
    print("Original list length:", len(main_list))
    print("Remove list length:", len(remove_list))
    print("New list length:", len(new_list))
    return new_list


def main():
    pass


if __name__ == "__main__":
    main()
