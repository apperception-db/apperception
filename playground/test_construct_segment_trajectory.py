from optimized_ingestion.stages.segment_trajectory.construct_segment_trajectory import *

if __name__ == '__main__':
    test_same_segment()
    test_wrong_start_same_segment()
    test_connected_segments()
    test_complete_story1()
    test_complete_story2()
    test_complete_story3()
    test_complete_story4()
    print('All tests passed!')