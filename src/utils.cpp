// for getting values bigger than the 32 bits that system() will return;
uint128_t calc(char * argv)
{
  uint128_t value;
  size_t len = 0;
  char * line = NULL;
  FILE * in;
  char cmd[256];

  sprintf(cmd, "calc %s | awk {'print $1'}", argv);

  in = popen(cmd, "r");
  getline(&line, &len, in);
  std::string s = line;

  value = string_to_u128(s);

  return value;
}

