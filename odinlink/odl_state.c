#include <stdio.h>
#include <odl_tb5/odl_tb5.h>
int main(void)
{
	for (int i = 0; i < 2; i++) {
		odl_tb5_t h;
		struct odl_tb5_peer_info info;
		if (odl_tb5_open(&h, i)) { printf("dev%d: open failed\n", i); continue; }
		if (odl_tb5_get_peer(h, &info))
			printf("dev%d: get_peer failed\n", i);
		else
			printf("dev%d: state=%s speed=%u\n", i,
			       odl_tb5_state_str(info.state), info.link_speed);
		odl_tb5_close(h);
	}
	return 0;
}
