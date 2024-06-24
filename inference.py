from anytext import aiy

def test1():
    """ 文本 + Mask """
    aiy_pipe = aiy.AiyAnyText()
    # test-generation
    imgs = aiy_pipe.text_generation('1 paper on the table, top-down perspective, with "中文" "Text" written on it using red pen',
                            'example_images/gen9.png',
                            n_steps=4,
                            seed=-1)
    for i, img in enumerate(imgs):
        img.save(f'test-2-{i}.png')

def test2():
    """ 文本 + 随机位置 """
    aiy_pipe = aiy.AiyAnyText()
    # test-generation
    imgs = aiy_pipe.text_generation('1 paper on the table, top-down perspective, with "中文" "Text" written on it using red pen',
                            draw_pos_path=None,
                            n_steps=4,
                            seed=-1)
    for i, img in enumerate(imgs):
        img.save(f'test-2-{i}.png')

def test3():
    """ 文本 + 图片编辑 """
    aiy_pipe = aiy.AiyAnyText()
    # test-generation
    ref = 'example_images/ref2.jpg'
    draw_pos_path = 'example_images/edit2.png'
    imgs = aiy_pipe.text_generation('"Close"',
                            draw_pos_path,
                            ori_image=ref,
                            mode='text-editing',
                            n_steps=4,
                            width=768,
                            height=768,
                            seed=-1)
    for i, img in enumerate(imgs):
        img.save(f'test-2-{i}.png')


if __name__ == '__main__':
    test3()
