"""4A D entry; the inherited C context adapter and W objective are unchanged."""
def main():
    from kit import sdft_verl as v
    from kit.a4_sdft_runner import A4TaskRunner
    v.check_pinned();v.main_ppo.TaskRunner=A4TaskRunner;v.main_ppo.main()

if __name__=='__main__':main()
